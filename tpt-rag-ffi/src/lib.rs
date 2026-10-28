use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;

use tpt_rag_core::{ChunkConfig, RAGLite, RagConfig, SearchResult};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: &str) {
    let message = to_cstring(message.to_string());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(message));
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Converts text to a C string. NUL bytes are removed so the rest of the text is kept.
fn to_cstring(mut text: String) -> CString {
    text.retain(|c| c != '\0');
    CString::new(text).unwrap_or_default()
}

/// Runs an entry point body. A panic is caught and turned into `on_panic`, so it never
/// unwinds across the C boundary. Each call clears the previous error first.
fn guard<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            set_last_error("internal error: panic inside tpt-rag");
            on_panic
        }
    }
}

fn invalid_arg(message: &str) -> TptRagError {
    set_last_error(message);
    TptRagError::TptRagErrInvalidArg
}

#[repr(C)]
pub enum TptRagError {
    TptRagOk = 0,
    TptRagErrInvalidArg = -1,
    TptRagErrIo = -2,
    TptRagErrSqlite = -3,
    TptRagErrOnnx = -4,
    TptRagErrEmptyDoc = -5,
    TptRagErrUnsupported = -6,
    TptRagErrInternal = -7,
}

#[repr(C)]
pub struct TptRagResult {
    pub score: f32,
    pub text: *mut c_char,
    pub source: *mut c_char,
    pub tags: *mut c_char,
}

pub struct TptRagHandle {
    pub(crate) rag: RAGLite,
}

/// Reads a NUL-terminated UTF-8 string argument.
///
/// # Safety
/// `ptr` must be null or point to a valid NUL-terminated string.
unsafe fn c_str_arg<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Reads `count` tag strings. Null or non-UTF-8 entries are skipped.
///
/// # Safety
/// `tags` must be null or point to `count` valid C string pointers.
unsafe fn collect_tags(tags: *const *const c_char, count: usize) -> Vec<String> {
    if tags.is_null() || count == 0 {
        return Vec::new();
    }
    std::slice::from_raw_parts(tags, count)
        .iter()
        .filter_map(|&t| c_str_arg(t).map(String::from))
        .collect()
}

fn map_error(e: tpt_rag_core::RagError) -> TptRagError {
    set_last_error(&e.to_string());
    match e {
        tpt_rag_core::RagError::Io(_) => TptRagError::TptRagErrIo,
        tpt_rag_core::RagError::Sqlite(_) => TptRagError::TptRagErrSqlite,
        tpt_rag_core::RagError::Onnx(_) => TptRagError::TptRagErrOnnx,
        tpt_rag_core::RagError::EmptyDocument(_) => TptRagError::TptRagErrEmptyDoc,
        tpt_rag_core::RagError::UnsupportedFormat(_) => TptRagError::TptRagErrUnsupported,
        tpt_rag_core::RagError::InvalidConfig(_) | tpt_rag_core::RagError::TooLarge { .. } => {
            TptRagError::TptRagErrInvalidArg
        }
        _ => TptRagError::TptRagErrInternal,
    }
}

/// Converts results into a boxed slice owned by the caller until `tpt_rag_free_results`.
/// Zero results give a null pointer and a count of 0.
///
/// # Safety
/// `results_out` and `count_out` must be valid for writes.
unsafe fn write_results(
    results: Vec<SearchResult>,
    results_out: *mut *mut TptRagResult,
    count_out: *mut usize,
) {
    let count = results.len();
    let items: Vec<TptRagResult> = results
        .into_iter()
        .map(|r| TptRagResult {
            score: r.score,
            text: to_cstring(r.text).into_raw(),
            source: to_cstring(r.source).into_raw(),
            tags: to_cstring(r.tags.join(",")).into_raw(),
        })
        .collect();

    *results_out = if count == 0 {
        ptr::null_mut()
    } else {
        Box::into_raw(items.into_boxed_slice()) as *mut TptRagResult
    };
    *count_out = count;
}

/// Frees a string returned by this crate. Null is accepted.
///
/// # Safety
/// `text` must be null or a pointer produced by `CString::into_raw` in this crate.
unsafe fn free_cstr(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}

/// Settings for `tpt_rag_create_with_config`. Start from `tpt_rag_default_config()` and change
/// the fields you need. `model_path` may be null to use the usual model search order.
#[repr(C)]
pub struct TptRagConfig {
    pub model_path: *const c_char,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub window_size: usize,
    pub overlap: usize,
    /// Write the index after this many changes. 0 turns this off.
    pub flush_every: usize,
    /// Re-embed every chunk when the database was built with another model.
    pub reembed_on_model_change: bool,
}

/// Returns the default settings. The `model_path` field is null.
#[no_mangle]
pub extern "C" fn tpt_rag_default_config() -> TptRagConfig {
    let defaults = RagConfig::default();
    TptRagConfig {
        model_path: ptr::null(),
        m: defaults.m,
        ef_construction: defaults.ef_construction,
        ef_search: defaults.ef_search,
        window_size: defaults.chunk.window_size,
        overlap: defaults.chunk.overlap,
        flush_every: defaults.flush_every,
        reembed_on_model_change: defaults.reembed_on_model_change,
    }
}

#[no_mangle]
/// # Safety
/// `path` must be null or a valid NUL-terminated string.
pub unsafe extern "C" fn tpt_rag_create(path: *const c_char) -> *mut TptRagHandle {
    guard(ptr::null_mut(), || {
        let Some(path) = c_str_arg(path) else {
            set_last_error("invalid argument: path is null or not UTF-8");
            return ptr::null_mut();
        };
        match RAGLite::open(PathBuf::from(path).as_path()) {
            Ok(rag) => Box::into_raw(Box::new(TptRagHandle { rag })),
            Err(e) => {
                map_error(e);
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
/// # Safety
/// `path` must be a valid NUL-terminated string; `config` must point to a valid `TptRagConfig`,
/// and its `model_path` must be null or a valid NUL-terminated string.
pub unsafe extern "C" fn tpt_rag_create_with_config(
    path: *const c_char,
    config: *const TptRagConfig,
) -> *mut TptRagHandle {
    guard(ptr::null_mut(), || {
        let Some(path) = c_str_arg(path) else {
            set_last_error("invalid argument: path is null or not UTF-8");
            return ptr::null_mut();
        };
        let Some(cfg) = config.as_ref() else {
            set_last_error("invalid argument: config is null");
            return ptr::null_mut();
        };
        let model_path = if cfg.model_path.is_null() {
            None
        } else {
            match c_str_arg(cfg.model_path) {
                Some(p) => Some(PathBuf::from(p)),
                None => {
                    set_last_error("invalid argument: model_path is not UTF-8");
                    return ptr::null_mut();
                }
            }
        };
        let rust_config = RagConfig {
            model_path,
            m: cfg.m,
            ef_construction: cfg.ef_construction,
            ef_search: cfg.ef_search,
            chunk: ChunkConfig {
                window_size: cfg.window_size,
                overlap: cfg.overlap,
                ..ChunkConfig::default()
            },
            flush_every: cfg.flush_every,
            reembed_on_model_change: cfg.reembed_on_model_change,
            ..RagConfig::default()
        };
        match RAGLite::open_with_config(PathBuf::from(path).as_path(), rust_config) {
            Ok(rag) => Box::into_raw(Box::new(TptRagHandle { rag })),
            Err(e) => {
                map_error(e);
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be null or a pointer from `tpt_rag_create` that has not been destroyed.
pub unsafe extern "C" fn tpt_rag_destroy(handle: *mut TptRagHandle) {
    guard((), || {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be a live handle from `tpt_rag_create`; string arguments must be valid NUL-terminated strings or null.
pub unsafe extern "C" fn tpt_rag_add_file(
    handle: *mut TptRagHandle,
    file_path: *const c_char,
    tags: *const *const c_char,
    tag_count: usize,
) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_mut() else {
            return invalid_arg("invalid argument: handle is null");
        };
        let Some(file_path) = c_str_arg(file_path) else {
            return invalid_arg("invalid argument: file path is null or not UTF-8");
        };
        let tag_vec = collect_tags(tags, tag_count);
        let tag_refs: Vec<&str> = tag_vec.iter().map(String::as_str).collect();

        match handle
            .rag
            .add_document(PathBuf::from(file_path).as_path(), &tag_refs)
        {
            Ok(_) => TptRagError::TptRagOk,
            Err(e) => map_error(e),
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be a live handle from `tpt_rag_create`; string arguments must be valid NUL-terminated strings or null.
pub unsafe extern "C" fn tpt_rag_add_text(
    handle: *mut TptRagHandle,
    text: *const c_char,
) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_mut() else {
            return invalid_arg("invalid argument: handle is null");
        };
        let Some(text) = c_str_arg(text) else {
            return invalid_arg("invalid argument: text is null or not UTF-8");
        };
        match handle
            .rag
            .add_text(text, &tpt_rag_core::Metadata::default())
        {
            Ok(_) => TptRagError::TptRagOk,
            Err(e) => map_error(e),
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be a live handle; `query_text` must be a valid string; `results_out` and `count_out` must be valid for writes.
pub unsafe extern "C" fn tpt_rag_query(
    handle: *mut TptRagHandle,
    query_text: *const c_char,
    top_k: usize,
    results_out: *mut *mut TptRagResult,
    count_out: *mut usize,
) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_ref() else {
            return invalid_arg("invalid argument: handle is null");
        };
        if results_out.is_null() || count_out.is_null() {
            return invalid_arg("invalid argument: output pointer is null");
        }
        let Some(query) = c_str_arg(query_text) else {
            return invalid_arg("invalid argument: query is null or not UTF-8");
        };
        match handle.rag.query(query, top_k) {
            Ok(results) => {
                write_results(results, results_out, count_out);
                TptRagError::TptRagOk
            }
            Err(e) => map_error(e),
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be a live handle; `query_text` must be a valid string; `results_out` and `count_out` must be valid for writes.
pub unsafe extern "C" fn tpt_rag_query_hybrid(
    handle: *mut TptRagHandle,
    query_text: *const c_char,
    top_k: usize,
    alpha: f32,
    results_out: *mut *mut TptRagResult,
    count_out: *mut usize,
) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_ref() else {
            return invalid_arg("invalid argument: handle is null");
        };
        if results_out.is_null() || count_out.is_null() {
            return invalid_arg("invalid argument: output pointer is null");
        }
        let Some(query) = c_str_arg(query_text) else {
            return invalid_arg("invalid argument: query is null or not UTF-8");
        };
        match handle.rag.query_hybrid(query, top_k, alpha) {
            Ok(results) => {
                write_results(results, results_out, count_out);
                TptRagError::TptRagOk
            }
            Err(e) => map_error(e),
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be a live handle; `query_text` must be a valid string; `results_out` and `count_out` must be valid for writes.
pub unsafe extern "C" fn tpt_rag_query_reranked(
    handle: *mut TptRagHandle,
    query_text: *const c_char,
    top_k: usize,
    rerank_top_n: usize,
    results_out: *mut *mut TptRagResult,
    count_out: *mut usize,
) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_ref() else {
            return invalid_arg("invalid argument: handle is null");
        };
        if results_out.is_null() || count_out.is_null() {
            return invalid_arg("invalid argument: output pointer is null");
        }
        let Some(query) = c_str_arg(query_text) else {
            return invalid_arg("invalid argument: query is null or not UTF-8");
        };
        match handle.rag.query_reranked(query, top_k, rerank_top_n) {
            Ok(results) => {
                write_results(results, results_out, count_out);
                TptRagError::TptRagOk
            }
            Err(e) => map_error(e),
        }
    })
}

#[no_mangle]
/// # Safety
/// `handle` must be a live handle; `source` must be a valid string; `deleted_out` may be null.
pub unsafe extern "C" fn tpt_rag_delete_document(
    handle: *mut TptRagHandle,
    source: *const c_char,
    deleted_out: *mut usize,
) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_ref() else {
            return invalid_arg("invalid argument: handle is null");
        };
        let Some(source) = c_str_arg(source) else {
            return invalid_arg("invalid argument: source is null or not UTF-8");
        };
        match handle.rag.delete_document(source) {
            Ok(deleted) => {
                if !deleted_out.is_null() {
                    *deleted_out = deleted;
                }
                TptRagError::TptRagOk
            }
            Err(e) => map_error(e),
        }
    })
}

/// Writes pending index changes to disk.
#[no_mangle]
/// # Safety
/// `handle` must be a live handle from `tpt_rag_create`.
pub unsafe extern "C" fn tpt_rag_flush(handle: *mut TptRagHandle) -> TptRagError {
    guard(TptRagError::TptRagErrInternal, || {
        let Some(handle) = handle.as_ref() else {
            return invalid_arg("invalid argument: handle is null");
        };
        match handle.rag.flush() {
            Ok(()) => TptRagError::TptRagOk,
            Err(e) => map_error(e),
        }
    })
}

#[no_mangle]
/// # Safety
/// `results` must be null or the array returned by a query, and `count` must match it.
pub unsafe extern "C" fn tpt_rag_free_results(results: *mut TptRagResult, count: usize) {
    guard((), || {
        if results.is_null() || count == 0 {
            return;
        }
        let owned = Box::from_raw(ptr::slice_from_raw_parts_mut(results, count));
        for result in owned.iter() {
            free_cstr(result.text);
            free_cstr(result.source);
            free_cstr(result.tags);
        }
    })
}

/// Returns the message for the most recent failed call on this thread, or null if the
/// last call succeeded. The pointer stays valid until the next call on this thread.
#[no_mangle]
pub extern "C" fn tpt_rag_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(ptr::null(), |m| m.as_ptr()))
}

/// Returns the library version as a static, NUL-terminated string.
#[no_mangle]
pub extern "C" fn tpt_rag_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}
