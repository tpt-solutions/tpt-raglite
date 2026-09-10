#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

enum class TptRagError {
  TptRagOk = 0,
  TptRagErrInvalidArg = -1,
  TptRagErrIo = -2,
  TptRagErrSqlite = -3,
  TptRagErrOnnx = -4,
  TptRagErrEmptyDoc = -5,
  TptRagErrUnsupported = -6,
  TptRagErrInternal = -7,
};

struct TptRagHandle;

struct TptRagResult {
  float score;
  char *text;
  char *source;
  char *tags;
};

extern "C" {

TptRagHandle *tpt_rag_create(const char *path);

void tpt_rag_destroy(TptRagHandle *handle);

TptRagError tpt_rag_add_file(TptRagHandle *handle,
                             const char *file_path,
                             const char *const *tags,
                             uintptr_t tag_count);

TptRagError tpt_rag_add_text(TptRagHandle *handle, const char *text);

TptRagError tpt_rag_query(TptRagHandle *handle,
                          const char *query_text,
                          uintptr_t top_k,
                          TptRagResult **results_out,
                          uintptr_t *count_out);

void tpt_rag_free_results(TptRagResult *results, uintptr_t count);

}  // extern "C"
