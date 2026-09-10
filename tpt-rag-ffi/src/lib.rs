use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use tpt_rag_core::RAGLite;

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

#[no_mangle]
pub extern "C" fn tpt_rag_create(path: *const c_char) -> *mut TptRagHandle {
    if path.is_null() {
        return ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match RAGLite::open(PathBuf::from(path_str).as_path()) {
        Ok(rag) => Box::into_raw(Box::new(TptRagHandle { rag })),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tpt_rag_destroy(handle: *mut TptRagHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[no_mangle]
pub extern "C" fn tpt_rag_add_file(
    handle: *mut TptRagHandle,
    file_path: *const c_char,
    tags: *const *const c_char,
    tag_count: usize,
) -> TptRagError {
    if handle.is_null() || file_path.is_null() {
        return TptRagError::TptRagErrInvalidArg;
    }
    let handle = unsafe { &mut *handle };
    let c_str = unsafe { CStr::from_ptr(file_path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return TptRagError::TptRagErrInvalidArg,
    };

    let tag_vec: Vec<String> = if tags.is_null() || tag_count == 0 {
        Vec::new()
    } else {
        let tags_slice = unsafe { std::slice::from_raw_parts(tags, tag_count) };
        tags_slice
            .iter()
            .filter_map(|t| {
                if t.is_null() {
                    None
                } else {
                    unsafe { CStr::from_ptr(*t) }
                        .to_str()
                        .ok()
                        .map(String::from)
                }
            })
            .collect()
    };

    let tag_refs: Vec<&str> = tag_vec.iter().map(|s| s.as_str()).collect();

    match handle.rag.add_document(PathBuf::from(path_str).as_path(), &tag_refs) {
        Ok(_) => TptRagError::TptRagOk,
        Err(e) => map_error(e),
    }
}

#[no_mangle]
pub extern "C" fn tpt_rag_add_text(
    handle: *mut TptRagHandle,
    text: *const c_char,
) -> TptRagError {
    if handle.is_null() || text.is_null() {
        return TptRagError::TptRagErrInvalidArg;
    }
    let handle = unsafe { &mut *handle };
    let c_str = unsafe { CStr::from_ptr(text) };
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return TptRagError::TptRagErrInvalidArg,
    };
    match handle.rag.add_text(text_str, &tpt_rag_core::Metadata::default()) {
        Ok(_) => TptRagError::TptRagOk,
        Err(e) => map_error(e),
    }
}

#[no_mangle]
pub extern "C" fn tpt_rag_query(
    handle: *mut TptRagHandle,
    query_text: *const c_char,
    top_k: usize,
    results_out: *mut *mut TptRagResult,
    count_out: *mut usize,
) -> TptRagError {
    if handle.is_null() || query_text.is_null() || results_out.is_null() || count_out.is_null() {
        return TptRagError::TptRagErrInvalidArg;
    }
    let handle = unsafe { &*handle };
    let c_str = unsafe { CStr::from_ptr(query_text) };
    let query = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return TptRagError::TptRagErrInvalidArg,
    };

    match handle.rag.query(query, top_k) {
        Ok(search_results) => {
            let count = search_results.len();
            let mut results: Vec<TptRagResult> = search_results
                .into_iter()
                .map(|r| {
                    let text = CString::new(r.text).unwrap_or_default();
                    let source = CString::new(r.source).unwrap_or_default();
                    let tags = CString::new(r.tags.join(",")).unwrap_or_default();
                    TptRagResult {
                        score: r.score,
                        text: text.into_raw(),
                        source: source.into_raw(),
                        tags: tags.into_raw(),
                    }
                })
                .collect();
            let ptr = results.as_mut_ptr();
            std::mem::forget(results);
            unsafe {
                *results_out = ptr;
                *count_out = count;
            }
            TptRagError::TptRagOk
        }
        Err(e) => map_error(e),
    }
}

#[no_mangle]
pub extern "C" fn tpt_rag_free_results(results: *mut TptRagResult, count: usize) {
    if results.is_null() || count == 0 {
        return;
    }
    let mut results = unsafe { Vec::from_raw_parts(results, count, count) };
    for r in results.iter_mut() {
        if !r.text.is_null() {
            unsafe { drop(CString::from_raw(r.text)) };
            r.text = ptr::null_mut();
        }
        if !r.source.is_null() {
            unsafe { drop(CString::from_raw(r.source)) };
            r.source = ptr::null_mut();
        }
        if !r.tags.is_null() {
            unsafe { drop(CString::from_raw(r.tags)) };
            r.tags = ptr::null_mut();
        }
    }
    drop(results);
}

fn map_error(e: tpt_rag_core::RagError) -> TptRagError {
    match e {
        tpt_rag_core::RagError::Io(_) => TptRagError::TptRagErrIo,
        tpt_rag_core::RagError::Sqlite(_) => TptRagError::TptRagErrSqlite,
        tpt_rag_core::RagError::Onnx(_) => TptRagError::TptRagErrOnnx,
        tpt_rag_core::RagError::EmptyDocument(_) => TptRagError::TptRagErrEmptyDoc,
        tpt_rag_core::RagError::UnsupportedFormat(_) => TptRagError::TptRagErrUnsupported,
        _ => TptRagError::TptRagErrInternal,
    }
}
