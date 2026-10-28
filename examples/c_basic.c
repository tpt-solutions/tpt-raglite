/* Minimal tpt-raglite C example. CI compiles and runs this file.
 *
 * Build (Linux, from the repository root, after `cargo build --release -p tpt-rag-ffi`):
 *   gcc -Wall -Wextra -I. examples/c_basic.c -Ltarget/release -ltpt_rag_ffi -o c_basic
 *   RAGLITE_MODEL=models/all-MiniLM-L6-v2.onnx LD_LIBRARY_PATH=target/release ./c_basic
 */
#include <stdio.h>
#include "tpt_raglite.h"

int main(void) {
    TptRagHandle *rag = tpt_rag_create("./my_c_db");
    if (!rag) {
        fprintf(stderr, "open failed: %s\n", tpt_rag_last_error());
        return 1;
    }

    if (tpt_rag_add_text(rag, "Jupiter is the largest planet in the solar system.") != TptRagOk) {
        fprintf(stderr, "ingest failed: %s\n", tpt_rag_last_error());
    }

    TptRagResult *results = NULL;
    size_t count = 0;
    if (tpt_rag_query(rag, "How big is Jupiter?", 5, &results, &count) == TptRagOk) {
        for (size_t i = 0; i < count; i++) {
            printf("[%.2f] %s\n", results[i].score, results[i].text);
        }
    }

    tpt_rag_free_results(results, count);
    tpt_rag_flush(rag);
    tpt_rag_destroy(rag);
    return 0;
}
