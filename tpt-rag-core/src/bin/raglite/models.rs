//! `raglite models pull`: downloads the embedding model, its vocabulary and the reranker.
//!
//! Each file is checked against a pinned SHA-256 hash, the same hashes as `scripts/fetch_model`.
//! A file that is already present and matches is kept. A download that does not match is
//! discarded, so a damaged or swapped file never reaches the models folder.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

/// One model file to download.
#[derive(Debug, Clone, Copy)]
pub struct ModelFile {
    pub url: &'static str,
    pub name: &'static str,
    pub sha256: &'static str,
}

/// The files `models pull` downloads. The hashes match `scripts/fetch_model.sh`.
pub const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        url: "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx",
        name: "all-MiniLM-L6-v2.onnx",
        sha256: "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
    },
    ModelFile {
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
        name: "vocab.txt",
        sha256: "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    },
    ModelFile {
        url: "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx",
        name: "ms-marco-MiniLM-L-6-v2.onnx",
        sha256: "e9d8ebf845c413e981c175bfe49a3bfa9b3dcce2a3ba54875ee5df5a58639fbe",
    },
];

/// What `pull` did with each file.
#[derive(Debug, Default)]
pub struct PullReport {
    pub downloaded: Vec<String>,
    pub kept: Vec<String>,
}

/// Lower-case hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Makes sure every file is in `dir`, downloading only the ones that are missing or differ.
/// Stops at the first failure. Files before it stay in place.
pub fn pull(dir: &Path, files: &[ModelFile]) -> Result<PullReport, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let mut report = PullReport::default();
    for file in files {
        if fetch(dir, file)? {
            report.downloaded.push(file.name.to_string());
        } else {
            report.kept.push(file.name.to_string());
        }
    }
    Ok(report)
}

/// Downloads one file if needed. Returns `true` when it downloaded the file.
fn fetch(dir: &Path, file: &ModelFile) -> Result<bool, String> {
    let target = dir.join(file.name);
    if let Ok(existing) = std::fs::read(&target) {
        if sha256_hex(&existing) == file.sha256 {
            return Ok(false);
        }
    }

    let response = ureq::get(file.url)
        .call()
        .map_err(|e| format!("download of {} failed: {e}", file.name))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("download of {} was interrupted: {e}", file.name))?;

    if sha256_hex(&bytes) != file.sha256 {
        return Err(format!(
            "{} does not match its pinned SHA-256 hash; nothing was written",
            file.name
        ));
    }
    // Write under a temporary name and rename, so an interrupted write never leaves a partial
    // file under the real name.
    let temporary = dir.join(format!(".{}.download", file.name));
    std::fs::write(&temporary, &bytes).map_err(|e| format!("cannot write {}: {e}", file.name))?;
    std::fs::rename(&temporary, &target).map_err(|e| format!("cannot save {}: {e}", file.name))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    /// Serves `body` on a local port for as long as the test process runs. Returns the URL and
    /// a counter of the requests served.
    fn serve(body: &'static [u8]) -> (String, Arc<AtomicUsize>) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let requests = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&requests);
        std::thread::spawn(move || {
            for request in server.incoming_requests() {
                counter.fetch_add(1, Ordering::SeqCst);
                let _ = request.respond(tiny_http::Response::from_data(body));
            }
        });
        (format!("http://127.0.0.1:{port}/model.onnx"), requests)
    }

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tpt_models_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn pinned_hashes_are_well_formed() {
        for file in MODEL_FILES {
            assert_eq!(file.sha256.len(), 64, "{}", file.name);
            assert!(file.sha256.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(
                file.url.starts_with("https://huggingface.co/"),
                "{}",
                file.url
            );
        }
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn downloads_a_file_whose_hash_matches_and_keeps_it_afterwards() {
        let body: &'static [u8] = b"model bytes";
        let (url, requests) = serve(body);
        let file = ModelFile {
            url: Box::leak(url.into_boxed_str()),
            name: "model.onnx",
            sha256: Box::leak(sha256_hex(body).into_boxed_str()),
        };
        let dir = temp_dir();

        let first = pull(&dir, &[file]).unwrap();
        assert_eq!(first.downloaded, ["model.onnx"]);
        assert_eq!(std::fs::read(dir.join("model.onnx")).unwrap(), body);
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        let second = pull(&dir, &[file]).unwrap();
        assert_eq!(second.kept, ["model.onnx"]);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "a matching file is not downloaded again"
        );
    }

    #[test]
    fn a_download_that_does_not_match_is_discarded() {
        let (url, _) = serve(b"unexpected contents");
        let file = ModelFile {
            url: Box::leak(url.into_boxed_str()),
            name: "model.onnx",
            sha256: Box::leak(sha256_hex(b"expected contents").into_boxed_str()),
        };
        let dir = temp_dir();

        let err = pull(&dir, &[file]).unwrap_err();
        assert!(err.contains("does not match"), "{err}");
        assert!(!dir.join("model.onnx").exists());
        let leftovers = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(leftovers, 0, "no temporary file is left behind");
    }

    #[test]
    fn a_file_that_differs_is_replaced() {
        let body: &'static [u8] = b"fresh model";
        let (url, requests) = serve(body);
        let file = ModelFile {
            url: Box::leak(url.into_boxed_str()),
            name: "model.onnx",
            sha256: Box::leak(sha256_hex(body).into_boxed_str()),
        };
        let dir = temp_dir();
        std::fs::write(dir.join("model.onnx"), b"old model").unwrap();

        assert_eq!(pull(&dir, &[file]).unwrap().downloaded, ["model.onnx"]);
        assert_eq!(std::fs::read(dir.join("model.onnx")).unwrap(), body);
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }
}
