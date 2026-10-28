"""The SQLite for AI: embeddable, zero-configuration RAG engine."""

from .tpt_raglite import (
    EmptyDocumentError,
    ModelError,
    RAGLite,
    RAGLiteError,
    SearchResult,
    UnsupportedFormatError,
    __version__,
)

__all__ = [
    "EmptyDocumentError",
    "ModelError",
    "RAGLite",
    "RAGLiteError",
    "SearchResult",
    "UnsupportedFormatError",
    "__version__",
]
