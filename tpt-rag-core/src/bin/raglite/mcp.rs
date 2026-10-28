//! `raglite serve --mcp`: the database as tools for an MCP client, over stdio.
//!
//! The client starts the process and writes one JSON-RPC message per line to stdin. Each
//! response is written as one line to stdout, so stdout carries nothing else. Tools run the
//! same routes as the HTTP API, so they validate their input the same way.
//!
//! Tools: `raglite_query`, `raglite_add_text`, `raglite_delete` and `raglite_stats`.

use std::io::{BufRead, Write};

use serde_json::{json, Value};
use tpt_rag_core::RAGLite;

use crate::serve::handle;

/// The MCP protocol version this server speaks.
const PROTOCOL_VERSION: &str = "2024-11-05";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

/// Reads requests from stdin and writes responses to stdout until stdin closes.
pub fn run(rag: &RAGLite) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = handle_line(rag, &line) {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

/// Answers one message. Returns `None` for notifications, which have no `id` and get no reply.
pub fn handle_line(rag: &RAGLite, line: &str) -> Option<Value> {
    let message: Value = match serde_json::from_str(line) {
        Ok(message) => message,
        Err(_) => return Some(error_response(Value::Null, PARSE_ERROR, "parse error")),
    };
    if !message.is_object() {
        return Some(error_response(
            Value::Null,
            INVALID_REQUEST,
            "request must be an object",
        ));
    }
    let id = message.get("id")?.clone();
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        return Some(error_response(id, INVALID_REQUEST, "request has no method"));
    };
    let params = message.get("params").cloned().unwrap_or(Value::Null);

    Some(match method {
        "initialize" => result_response(id, initialize()),
        "ping" => result_response(id, json!({})),
        "tools/list" => result_response(id, json!({ "tools": tools() })),
        "tools/call" => match call_tool(rag, &params) {
            Ok(result) => result_response(id, result),
            Err((code, message)) => error_response(id, code, &message),
        },
        _ => error_response(id, METHOD_NOT_FOUND, "method not found"),
    })
}

fn result_response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

fn initialize() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "raglite", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// Runs a tool. A tool that fails returns a result with `isError` set, so the model can see
/// the message. Only an unknown tool or a malformed call is a JSON-RPC error.
fn call_tool(rag: &RAGLite, params: &Value) -> Result<Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((INVALID_PARAMS, "tools/call needs a tool name".to_string()))?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let body = arguments.to_string();

    let (status, value) = match name {
        "raglite_query" => handle(rag, "POST", "/query", body.as_bytes()),
        "raglite_add_text" => handle(rag, "POST", "/text", body.as_bytes()),
        "raglite_delete" => handle(rag, "POST", "/delete", body.as_bytes()),
        "raglite_stats" => handle(rag, "GET", "/stats", b""),
        other => return Err((INVALID_PARAMS, format!("unknown tool {other:?}"))),
    };
    Ok(json!({
        "content": [{ "type": "text", "text": value.to_string() }],
        "isError": status != 200,
    }))
}

fn tools() -> Value {
    json!([
        {
            "name": "raglite_query",
            "description": "Search the knowledge base for passages relevant to a question. Each result has its text, source, section or page, byte offsets, and a score.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The question or search text." },
                    "top_k": { "type": "integer", "minimum": 1, "maximum": 1000, "description": "Number of results (default 5)." },
                    "mode": { "type": "string", "enum": ["vector", "hybrid", "rerank", "diverse"], "description": "Search method (default vector)." },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Vector mode: only chunks with every tag." },
                    "source_prefix": { "type": "string", "description": "Vector mode: only chunks whose source starts with this." },
                    "since": { "type": "integer", "description": "Vector mode: only chunks ingested at or after this Unix time." },
                    "alpha": { "type": "number", "description": "Hybrid mode: vector weight from 0 to 1 (default 0.5)." },
                    "rerank_top_n": { "type": "integer", "description": "Rerank mode: candidates to rerank (default 20)." },
                    "relevance_weight": { "type": "number", "description": "Diverse mode: 1 is plain relevance, 0 is the most varied (default 0.5)." }
                },
                "required": ["text"]
            }
        },
        {
            "name": "raglite_add_text",
            "description": "Add a passage to the knowledge base.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The passage to store." },
                    "source": { "type": "string", "description": "Where the passage came from, such as a document name." },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["text"]
            }
        },
        {
            "name": "raglite_delete",
            "description": "Remove every passage that came from a source.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": { "type": "string", "description": "The source to remove, as given when it was added." }
                },
                "required": ["source"]
            }
        },
        {
            "name": "raglite_stats",
            "description": "Count the passages in the knowledge base and report the model in use.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

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

    fn call(rag: &RAGLite, name: &str, arguments: Value) -> Value {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": { "name": name, "arguments": arguments },
        });
        let response = handle_line(rag, &request.to_string()).expect("a reply");
        response["result"].clone()
    }

    #[test]
    fn initialize_and_list_tools() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        let reply = handle_line(
            &rag,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
        )
        .unwrap();
        assert_eq!(reply["id"], 1);
        assert_eq!(reply["result"]["serverInfo"]["name"], "raglite");
        assert!(reply["result"]["capabilities"]["tools"].is_object());

        let tools = handle_line(&rag, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#).unwrap();
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "raglite_query",
                "raglite_add_text",
                "raglite_delete",
                "raglite_stats"
            ]
        );
    }

    #[test]
    fn notifications_get_no_reply() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        assert!(handle_line(
            &rag,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#
        )
        .is_none());
    }

    #[test]
    fn add_then_query_through_tools() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        let added = call(
            &rag,
            "raglite_add_text",
            json!({ "text": "Backups run every night at two.", "source": "ops" }),
        );
        assert_eq!(added["isError"], false);

        let found = call(
            &rag,
            "raglite_query",
            json!({ "text": "when do backups run", "top_k": 1 }),
        );
        assert_eq!(found["isError"], false);
        let text = found["content"][0]["text"].as_str().unwrap();
        let payload: Value = serde_json::from_str(text).unwrap();
        assert_eq!(payload["results"][0]["source"], "ops");
    }

    #[test]
    fn tool_failures_are_reported_inside_the_result() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        let result = call(&rag, "raglite_query", json!({ "top_k": 2 }));
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("text"));
    }

    #[test]
    fn protocol_errors_are_json_rpc_errors() {
        let Some((_dir, rag)) = open_test_db() else {
            return;
        };
        let parse = handle_line(&rag, "{not json").unwrap();
        assert_eq!(parse["error"]["code"], PARSE_ERROR);
        assert!(parse["id"].is_null());

        let unknown = handle_line(&rag, r#"{"jsonrpc":"2.0","id":3,"method":"nope"}"#).unwrap();
        assert_eq!(unknown["error"]["code"], METHOD_NOT_FOUND);

        let no_tool = handle_line(
            &rag,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"raglite_explode"}}"#,
        )
        .unwrap();
        assert_eq!(no_tool["error"]["code"], INVALID_PARAMS);
    }
}
