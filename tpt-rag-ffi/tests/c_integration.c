/* Integration test for the C ABI. Exits 0 when every check passes.
 *
 * Usage: c_integration <empty-directory>
 * Needs RAGLITE_MODEL to point at all-MiniLM-L6-v2.onnx (vocab.txt beside it).
 * Run by CI through tpt-rag-ffi/tests/run_c_integration.sh.
 */
#include <stdio.h>
#include <string.h>
#include "tpt_raglite.h"

static int failures = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            const char *err = tpt_rag_last_error();                              \
            fprintf(stderr, "FAIL line %d: %s (last error: %s)\n", __LINE__,     \
                    what, err ? err : "none");                                   \
            failures++;                                                          \
        }                                                                        \
    } while (0)

static int write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return 0;
    }
    fputs(content, f);
    fclose(f);
    return 1;
}

static void check_top_text_contains(TptRagResult *results, size_t count, const char *needle,
                                    const char *what) {
    CHECK(count > 0, what);
    CHECK(results != NULL, what);
    if (count > 0 && results != NULL) {
        CHECK(strstr(results[0].text, needle) != NULL, what);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <empty-directory>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    char db[1024];
    char file[1024];
    snprintf(db, sizeof db, "%s/db", dir);
    snprintf(file, sizeof file, "%s/mars.txt", dir);

    /* Invalid arguments fail with a message and a null handle. */
    CHECK(tpt_rag_create(NULL) == NULL, "create with null path returns null");
    CHECK(tpt_rag_last_error() != NULL, "create with null path sets an error");
    CHECK(tpt_rag_flush(NULL) == TptRagErrInvalidArg, "flush with null handle");

    /* A zero window size is rejected by the config check. */
    TptRagConfig cfg = tpt_rag_default_config();
    cfg.overlap = cfg.window_size;
    CHECK(tpt_rag_create_with_config(db, &cfg) == NULL, "config with overlap >= window");

    TptRagHandle *rag = tpt_rag_create(db);
    CHECK(rag != NULL, "create database");
    if (!rag) {
        fprintf(stderr, "cannot continue without a database\n");
        return 1;
    }

    CHECK(tpt_rag_add_text(rag, "Jupiter is the largest planet in the solar system.") == TptRagOk,
          "add_text");
    CHECK(tpt_rag_add_text(rag, "Tomato plants need full sun and regular watering.") == TptRagOk,
          "add_text second");

    CHECK(write_file(file, "Mars is the fourth planet from the sun and is called the red planet.\n"),
          "write fixture file");
    const char *const tags[] = { "science" };
    CHECK(tpt_rag_add_file(rag, file, tags, 1) == TptRagOk, "add_file");

    const char *bad = "unsupported.xyz";
    CHECK(tpt_rag_add_file(rag, bad, NULL, 0) == TptRagErrUnsupported, "unsupported extension");

    TptRagResult *results = NULL;
    size_t count = 0;
    CHECK(tpt_rag_query(rag, "largest planet", 1, &results, &count) == TptRagOk, "query");
    check_top_text_contains(results, count, "Jupiter", "top result is Jupiter");
    tpt_rag_free_results(results, count);

    results = NULL;
    count = 0;
    CHECK(tpt_rag_query(rag, "which planet is red", 1, &results, &count) == TptRagOk,
          "query for file content");
    if (count > 0 && results != NULL) {
        CHECK(strstr(results[0].source, "mars.txt") != NULL, "source names the file");
        CHECK(strcmp(results[0].tags, "science") == 0, "tags come back");
    }
    tpt_rag_free_results(results, count);

    results = NULL;
    count = 0;
    CHECK(tpt_rag_query_hybrid(rag, "Tomato", 2, 0.5f, &results, &count) == TptRagOk,
          "hybrid query");
    check_top_text_contains(results, count, "Tomato", "hybrid top result is Tomato");
    if (count > 0 && results != NULL) {
        CHECK(results[0].score > 0.0f && results[0].score <= 1.0f, "hybrid score in (0, 1]");
    }
    tpt_rag_free_results(results, count);

    size_t deleted = 99;
    CHECK(tpt_rag_delete_document(rag, file, &deleted) == TptRagOk, "delete_document");
    CHECK(deleted == 1, "delete removes one chunk");

    CHECK(tpt_rag_query(rag, NULL, 1, &results, &count) == TptRagErrInvalidArg,
          "query with null text is rejected");
    CHECK(tpt_rag_flush(rag) == TptRagOk, "flush");
    tpt_rag_destroy(rag);

    /* Reopening sees what was written, and the deleted file stays deleted. */
    rag = tpt_rag_create(db);
    CHECK(rag != NULL, "reopen database");
    if (rag) {
        results = NULL;
        count = 0;
        CHECK(tpt_rag_query(rag, "largest planet", 1, &results, &count) == TptRagOk,
              "query after reopen");
        check_top_text_contains(results, count, "Jupiter", "Jupiter survives reopen");
        tpt_rag_free_results(results, count);

        results = NULL;
        count = 0;
        CHECK(tpt_rag_query(rag, "red planet", 5, &results, &count) == TptRagOk,
              "query for deleted file");
        for (size_t i = 0; results != NULL && i < count; i++) {
            CHECK(strstr(results[i].source, "mars.txt") == NULL, "deleted file is gone");
        }
        tpt_rag_free_results(results, count);
        tpt_rag_destroy(rag);
    }

    if (failures == 0) {
        printf("all C integration checks passed\n");
        return 0;
    }
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
}
