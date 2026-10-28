//! `raglite serve`: a small JSON API over HTTP.
//!
//! Requests are handled one at a time, so the database handle is never shared between
//! threads. The API has no authentication, so it binds to a loopback address unless the
//! caller passes `allow_remote`. Requests cannot read files from the server's disk: text is
//! sent in the request body, and documents are removed by source.
//!
//! Routes:
//!
//! - `GET /health` returns `{"status": "ok"}`.
//! - `GET /stats` returns chunk and vector counts, the index size, and the model.
//! - `POST /query` takes `{"text", "top_k", "mode", "tags", "source_prefix", "since"}` and
//!   returns `{"results": [...]}`. `mode` is `vector` (the default), `hybrid` (with `alpha`),
//!   `rerank` (with `rerank_top_n`), or `diverse` (with `relevance_weight`). Filters work only
//!   in `vector` mode, as they do in the CLI.
//! - `POST /text` takes `{"text", "source", "tags"}` and returns `{"ok": true}`.
//! - `POST /delete` takes `{"source"}` and returns `{"removed": n}`.
//!
//! Errors return `{"error": "..."}` with a 4xx or 5xx status.

use std::io::Read;
use std::net::SocketAddr;

use serde_json::{json, Value};
use tpt_rag_core::{Metadata, QueryFilter, RAGLite, RagError, SearchResult};

/// Largest request body accepted, in bytes.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Largest `top_k` a query may ask for.
const MAX_TOP_K: usize = 1000;

/// Serves the API on `addr` until the process is stopped.
pub fn serve(
    rag: &RAGLite,
    addr: &str,
    allow_remote: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket: SocketAddr = addr.parse()?;
    if !socket.ip().is_loopback() && !allow_remote {
        return Err(format!(
            "refusing to serve on {addr}: the API has no authentication. Bind to a loopback \
             address, or pass --allow-remote to accept that risk."
        )
        .into());
    }

    let server =
        tiny_http::Server::http(socket).map_err(|e| format!("could not listen on {addr}: {e}"))?;
    println!("listening on http://{socket} (Ctrl-C to stop)");
    for mut request in server.incoming_requests() {
        let (status, value) = match read_body(&mut request) {
            Ok(body) => handle(rag, request.method().as_str(), request.url(), &body),
            Err(status) => (status, error_value("request body is too large")),
        };
        let response = tiny_http::Response::from_string(value.to_string())
            .with_status_code(status)
            .with_header(
                tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                    .expect("valid header"),
            );
        if let Err(e) = request.respond(response) {
            eprintln!("could not send response: {e}");
        }
    }
    Ok(())
}

/// Reads the request body, or returns 413 when it is over the limit.
fn read_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, u16> {
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|_| 400u16)?;
    if body.len() > MAX_BODY_BYTES {
        return Err(413);
    }
    Ok(body)
}

/// A failed request: the status code and the message for the client.
#[derive(Debug)]
struct Failure(u16, String);

type Outcome = Result<Value, Failure>;

fn bad_request(message: impl Into<String>) -> Failure {
    Failure(400, message.into())
}

fn server_error(e: RagError) -> Failure {
    Failure(500, e.to_string())
}

fn error_value(message: &str) -> Value {
    json!({ "error": message })
}

/// Routes one request. Returns the status code and the JSON body.
pub fn handle(rag: &RAGLite, method: &str, url: &str, body: &[u8]) -> (u16, Value) {
    let path = url.split('?').next().unwrap_or("");
    let outcome = match (method, path) {
        ("GET", "/health") => Ok(json!({ "status": "ok" })),
        ("GET", "/stats") => stats(rag),
        ("POST", "/query") => json_object(body).and_then(|b| query(rag, &b)),
        ("POST", "/text") => json_object(body).and_then(|b| add_text(rag, &b)),
        ("POST", "/delete") => json_object(body).and_then(|b| delete(rag, &b)),
        (_, "/health" | "/stats" | "/query" | "/text" | "/delete") => {
            Err(Failure(405, "method not allowed".to_string()))
        }
        _ => Err(Failure(404, "no such route".to_string())),
    };
    match outcome {
        Ok(value) => (200, value),
        Err(Failure(status, message)) => (status, error_value(&message)),
    }
}

fn json_object(body: &[u8]) -> Result<Value, Failure> {
    let value: Value =
        serde_json::from_slice(body).map_err(|_| bad_request("request body is not valid JSON"))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(bad_request("request body must be a JSON object"))
    }
}

fn stats(rag: &RAGLite) -> Outcome {
    let stats = rag.stats().map_err(server_error)?;
    Ok(json!({
        "chunks": stats.chunks,
        "vectors": stats.vectors,
        "index_bytes": stats.index_bytes,
        "model": stats.model,
    }))
}

fn query(rag: &RAGLite, body: &Value) -> Outcome {
    let text = required_str(body, "text")?;
    let top_k = optional_usize(body, "top_k")?.unwrap_or(5);
    if top_k > MAX_TOP_K {
        return Err(bad_request(format!("top_k must be at most {MAX_TOP_K}")));
    }
    let tags = string_list(body, "tags")?;
    let source_prefix = optional_str(body, "source_prefix")?.map(str::to_string);
    let created_since = optional_u64(body, "since")?;
    let filter = QueryFilter {
        tags,
        source_prefix,
        created_since,
    };
    let filtered =
        !filter.tags.is_empty() || filter.source_prefix.is_some() || filter.created_since.is_some();

    let mode = optional_str(body, "mode")?.unwrap_or("vector");
    let results = match mode {
        "vector" if filtered => rag.query_filtered(text, top_k, &filter),
        "vector" => rag.query(text, top_k),
        "hybrid" | "rerank" | "diverse" if filtered => {
            return Err(bad_request(format!(
                "mode {mode:?} cannot be combined with tags, source_prefix or since"
            )))
        }
        "hybrid" => {
            let alpha = optional_f32(body, "alpha")?.unwrap_or(0.5);
            rag.query_hybrid(text, top_k, alpha)
        }
        "rerank" => {
            let rerank_top_n = optional_usize(body, "rerank_top_n")?.unwrap_or(20);
            rag.query_reranked(text, top_k, rerank_top_n)
        }
        "diverse" => {
            let weight = optional_f32(body, "relevance_weight")?.unwrap_or(0.5);
            rag.query_diverse(text, top_k, weight)
        }
        other => return Err(bad_request(format!("unknown mode {other:?}"))),
    }
    .map_err(server_error)?;

    Ok(json!({
        "results": results.iter().map(result_json).collect::<Vec<Value>>(),
    }))
}

fn add_text(rag: &RAGLite, body: &Value) -> Outcome {
    let text = required_str(body, "text")?;
    let metadata = Metadata {
        source: optional_str(body, "source")?.map(str::to_string),
        tags: string_list(body, "tags")?,
    };
    rag.add_text(text, &metadata).map_err(server_error)?;
    Ok(json!({ "ok": true }))
}

fn delete(rag: &RAGLite, body: &Value) -> Outcome {
    let source = required_str(body, "source")?;
    let removed = rag.delete_document(source).map_err(server_error)?;
    Ok(json!({ "removed": removed }))
}

fn result_json(r: &SearchResult) -> Value {
    json!({
        "score": r.score,
        "text": r.text,
        "source": r.source,
        "tags": r.tags,
        "chunk_id": r.chunk_id,
        "created_at": r.created_at,
        "start_offset": r.start_offset,
        "end_offset": r.end_offset,
        "section": r.section,
    })
}

fn required_str<'a>(body: &'a Value, name: &str) -> Result<&'a str, Failure> {
    body.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request(format!("{name} must be a string")))
}

/// A string field that may be absent or null.
fn optional_str<'a>(body: &'a Value, name: &str) -> Result<Option<&'a str>, Failure> {
    match body.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| bad_request(format!("{name} must be a string"))),
    }
}

fn optional_u64(body: &Value, name: &str) -> Result<Option<u64>, Failure> {
    match body.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| bad_request(format!("{name} must be a non-negative integer"))),
    }
}

fn optional_usize(body: &Value, name: &str) -> Result<Option<usize>, Failure> {
    match optional_u64(body, name)? {
        None => Ok(None),
        Some(n) => usize::try_from(n)
            .map(Some)
            .map_err(|_| bad_request(format!("{name} is too large"))),
    }
}

fn optional_f32(body: &Value, name: &str) -> Result<Option<f32>, Failure> {
    match body.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(|n| Some(n as f32))
            .ok_or_else(|| bad_request(format!("{name} must be a number"))),
    }
}

fn string_list(body: &Value, name: &str) -> Result<Vec<String>, Failure> {
    match body.get(name) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad_request(format!("{name} must be a list of strings")))
            })
            .collect(),
        Some(_) => Err(bad_request(format!("{name} must be a list of strings"))),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// Opens a fresh database with the real model. Skips (returns `None`) when the model is not
    /// downloaded, like the integration tests do.
    fn open_test_db() -> Option<(tempfile::TempDir, RAGLite)> {
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/all-MiniLM-L6-v2.onnx");
        if !model.exists() {
            eprintln!(
                "skipping: {} not found, run scripts/fetch_model",
                model.display()
            );
            return None;
        }
        let dir = tempfile::tempdir().ok()?;
        let rag = RAGLite::open_with_model(&dir.path().join("db"), &model).ok()?;
        Some((dir, rag))
    }

    fn post(rag: &RAGLite, path: &str, body: &str) -> (u16, Value) {
        handle(rag, "POST", path, body.as_bytes())
    }

    #[test]
    fn health_and_routing() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        assert_eq!(
            handle(&rag, "GET", "/health", b""),
            (200, json!({ "status": "ok" }))
        );
        assert_eq!(handle(&rag, "GET", "/nope", b"").0, 404);
        assert_eq!(handle(&rag, "GET", "/query", b"").0, 405);
        assert_eq!(handle(&rag, "GET", "/health?verbose=1", b"").0, 200);
    }

    #[test]
    fn text_is_searchable_and_can_be_deleted() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        let (status, _) = post(
            &rag,
            "/text",
            r#"{"text": "Jupiter is the largest planet.", "source": "space"}"#,
        );
        assert_eq!(status, 200);

        let (status, body) = post(&rag, "/query", r#"{"text": "largest planet", "top_k": 1}"#);
        assert_eq!(status, 200);
        let first = &body["results"][0];
        assert_eq!(first["source"], "space");
        assert!(first["section"].is_null());
        assert!(first["score"].as_f64().unwrap() > 0.0);

        let (status, body) = post(&rag, "/stats", "");
        assert_eq!(status, 405, "stats is a GET route: {body}");
        assert_eq!(handle(&rag, "GET", "/stats", b"").1["chunks"], 1);

        assert_eq!(
            post(&rag, "/delete", r#"{"source": "space"}"#).1["removed"],
            1
        );
    }

    #[test]
    fn bad_requests_get_a_400_with_a_message() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        for body in [
            "not json",
            "[1, 2]",
            r#"{"text": 5}"#,
            r#"{"text": "a", "top_k": -1}"#,
        ] {
            let (status, value) = post(&rag, "/query", body);
            assert_eq!(status, 400, "{body}");
            assert!(value["error"].as_str().is_some());
        }
        assert_eq!(
            post(&rag, "/query", r#"{"text": "a", "top_k": 5000}"#).0,
            400
        );
        assert_eq!(
            post(&rag, "/query", r#"{"text": "a", "mode": "fuzzy"}"#).0,
            400
        );
    }

    #[test]
    fn filters_cannot_be_combined_with_hybrid_rerank_or_diverse() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        for mode in ["hybrid", "rerank", "diverse"] {
            let body = format!(r#"{{"text": "a", "mode": "{mode}", "tags": ["x"]}}"#);
            let (status, value) = post(&rag, "/query", &body);
            assert_eq!(status, 400, "{mode}");
            assert!(value["error"].as_str().unwrap().contains(mode));
        }
    }

    #[test]
    fn remote_addresses_are_refused_without_allow_remote() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        let err = serve(&rag, "0.0.0.0:0", false).unwrap_err();
        assert!(err.to_string().contains("refusing to serve"), "{err}");
    }
}
