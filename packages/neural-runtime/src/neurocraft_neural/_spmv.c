/*
 * neurocraft _spmv.c -- fused CSR sparse-matrix/dense-vector product, float64.
 *
 * SCOPE
 *   One entry point computes y[r] = sum_{k in [indptr[r], indptr[r+1])}
 *   values[k] * x[indices[k]] for every row r, in a single fused pass. No
 *   intermediate product array is materialised, so an empty row (indptr[r] ==
 *   indptr[r+1]) leaves the accumulator at its initial +0.0 and stores exactly
 *   0.0. That is the whole reason this kernel exists rather than a three-pass
 *   gather/multiply/segmented-reduce: the segmented-reduce formulation returns
 *   a foreign element for a zero-length segment instead of zero.
 *
 * NUMERICAL CONTRACT (load-bearing -- do not "optimise" any of it)
 *   * float64 only. Every accumulator is a single scalar `double`.
 *   * Each row is summed strictly left to right in stored column order by
 *     exactly one thread. Rows are never split across threads, so the
 *     summation order -- and therefore every bit of the result -- is
 *     independent of the thread count.
 *   * The translation unit must be compiled with -ffp-contract=off (GNU
 *     drivers) or /fp:strict (MSVC/clang-cl drivers) and
 *     without -ffast-math/-Ofast. clang defaults -ffp-contract to `on` for
 *     C99 and later, which silently fuses `s + values[k] * x[...]` into an
 *     FMA. FMA is *more* accurate, which is exactly why it must not appear by
 *     accident: it would make results depend on whether the target has an FMA
 *     unit, breaking bit-level replay between machines. With contraction off
 *     and no fast-math, IEEE-754 binary64 `*` and `+` are exactly specified
 *     and no conforming compiler may reorder the reduction. Windows uses
 *     native Win32 threads and cdecl DLL exports; POSIX uses pthreads.
 *     Both paths run the same row arithmetic and bounds checks.
 *   * The bounds checks added below are integer comparisons on values that are
 *     never operands of the reduction. They cannot change any stored bit; a
 *     conforming run either produces the identical result or refuses.
 *
 * STRUCTURAL PRECONDITIONS ARE CHECKED, NOT ASSUMED (ABI 2)
 *   Earlier revisions documented `indices[k] < n_cols` and a non-decreasing
 *   `indptr` as caller preconditions validated once in Python, and performed no
 *   per-element checking. That is unsound for this backend: the Python layer
 *   validates at *view construction*, while the amortised integrity checkpoint
 *   that is supposed to catch later corruption runs *after* the kernel has
 *   already dereferenced the buffer for that step. A single out-of-range index
 *   written after construction therefore reached this loop first and produced
 *   an out-of-bounds read -- undefined behaviour, observed as SIGSEGV.
 *
 *   This kernel now validates, per row, that the CSR offsets lie in
 *   [0, nnz] and are non-decreasing, and per element that the column index is
 *   below n_cols. A violation stops the block and the call returns
 *   NEUROCRAFT_SPMV_EDOMAIN, which the Python layer turns into a
 *   SpmvKernelError. Corruption of the index or offset buffers is therefore
 *   *contained*, exactly as the reference engine contains it with an
 *   IndexError, instead of being undefined behaviour.
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define NEUROCRAFT_SPMV_EXPORT __declspec(dllexport)
#define NEUROCRAFT_SPMV_CALL __cdecl
typedef HANDLE neurocraft_spmv_thread;
#else
#include <pthread.h>
#define NEUROCRAFT_SPMV_EXPORT
#define NEUROCRAFT_SPMV_CALL
typedef pthread_t neurocraft_spmv_thread;
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define NEUROCRAFT_SPMV_ABI_VERSION 2
#define NEUROCRAFT_SPMV_MAX_THREADS 64

/* Error codes returned by neurocraft_spmv_csr_f64. */
#define NEUROCRAFT_SPMV_OK 0
#define NEUROCRAFT_SPMV_EINVAL (-1)
#define NEUROCRAFT_SPMV_ENOMEM (-2)
#define NEUROCRAFT_SPMV_EDOMAIN (-3)

typedef struct {
    const int64_t *indptr;
    const uint32_t *indices;
    const double *values;
    const double *x;
    double *y;
    int64_t row_begin;
    int64_t row_end;
    int64_t nnz;
    uint64_t n_cols;
    int32_t status;
} neurocraft_spmv_block;

/*
 * The fused inner loop. `sum` starts at +0.0; an empty row therefore stores
 * exactly +0.0 with no branch and no special case. The two range tests are
 * loop-invariant-compare / never-taken-branch shaped and do not participate in
 * the floating-point reduction.
 */
static void neurocraft_spmv_run_block(neurocraft_spmv_block *block) {
    const int64_t *const indptr = block->indptr;
    const uint32_t *const indices = block->indices;
    const double *const values = block->values;
    const double *const x = block->x;
    double *const y = block->y;
    const uint64_t n_cols = block->n_cols;
    const int64_t nnz = block->nnz;
    int64_t row;

    for (row = block->row_begin; row < block->row_end; ++row) {
        const int64_t start = indptr[row];
        const int64_t stop = indptr[row + 1];
        double sum = 0.0;
        int64_t k;
        if (start < 0 || stop < start || stop > nnz) {
            block->status = NEUROCRAFT_SPMV_EDOMAIN;
            return;
        }
        for (k = start; k < stop; ++k) {
            const uint32_t column = indices[k];
            if ((uint64_t)column >= n_cols) {
                block->status = NEUROCRAFT_SPMV_EDOMAIN;
                return;
            }
            sum += values[k] * x[column];
        }
        y[row] = sum;
    }
}

#ifdef _WIN32
static DWORD WINAPI neurocraft_spmv_thread_entry(LPVOID argument) {
    neurocraft_spmv_run_block((neurocraft_spmv_block *)argument);
    return 0;
}

static int neurocraft_spmv_thread_start(neurocraft_spmv_thread *thread, void *argument) {
    *thread = CreateThread(NULL, 0, neurocraft_spmv_thread_entry, argument, 0, NULL);
    return *thread == NULL ? -1 : 0;
}

static void neurocraft_spmv_thread_join(neurocraft_spmv_thread thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
#else
static void *neurocraft_spmv_thread_entry(void *argument) {
    neurocraft_spmv_run_block((neurocraft_spmv_block *)argument);
    return NULL;
}

static int neurocraft_spmv_thread_start(neurocraft_spmv_thread *thread, void *argument) {
    return pthread_create(thread, NULL, neurocraft_spmv_thread_entry, argument);
}

static void neurocraft_spmv_thread_join(neurocraft_spmv_thread thread) {
    pthread_join(thread, NULL);
}
#endif

/*
 * Contiguous row blocks balanced by nonzero count. Deterministic given
 * (indptr, n_rows, n_blocks): boundary b is the first row whose prefix
 * nonzero count reaches ceil-free target (nnz * b) / n_blocks, found by binary
 * search. Because the partition only decides *which* thread sums a row and
 * never *how*, changing n_blocks cannot change any stored value.
 */
static void neurocraft_spmv_partition(const int64_t *indptr, int64_t n_rows, int64_t nnz,
                                      int32_t n_blocks, int64_t *bounds) {
    int32_t block;

    bounds[0] = 0;
    bounds[n_blocks] = n_rows;
    for (block = 1; block < n_blocks; ++block) {
        const int64_t target = (int64_t)((double)nnz * (double)block / (double)n_blocks);
        int64_t low = bounds[block - 1];
        int64_t high = n_rows;
        while (low < high) {
            const int64_t middle = low + (high - low) / 2;
            if (indptr[middle] < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        bounds[block] = low;
    }
}

NEUROCRAFT_SPMV_EXPORT int32_t NEUROCRAFT_SPMV_CALL neurocraft_spmv_abi_version(void) {
    return NEUROCRAFT_SPMV_ABI_VERSION;
}

NEUROCRAFT_SPMV_EXPORT int32_t NEUROCRAFT_SPMV_CALL neurocraft_spmv_max_threads(void) {
    return NEUROCRAFT_SPMV_MAX_THREADS;
}

/*
 * y[0:n_rows] = A @ x, A given in CSR with nnz stored entries. Returns
 * NEUROCRAFT_SPMV_OK on success, NEUROCRAFT_SPMV_EDOMAIN if the offset or
 * index buffers violate their declared range invariants.
 */
NEUROCRAFT_SPMV_EXPORT int32_t NEUROCRAFT_SPMV_CALL
neurocraft_spmv_csr_f64(const int64_t *indptr, const uint32_t *indices,
                                const double *values, const double *x, double *y, int64_t n_rows,
                                int64_t n_cols, int64_t nnz, int32_t n_threads) {
    neurocraft_spmv_block single;
    neurocraft_spmv_block *blocks = NULL;
    neurocraft_spmv_thread *threads = NULL;
    int64_t *bounds = NULL;
    int32_t started = 0;
    int32_t status = NEUROCRAFT_SPMV_OK;
    int32_t block;

    if (indptr == NULL || indices == NULL || values == NULL || x == NULL || y == NULL) {
        return NEUROCRAFT_SPMV_EINVAL;
    }
    if (n_rows < 0 || n_cols < 0 || nnz < 0 || n_threads < 1 ||
        n_threads > NEUROCRAFT_SPMV_MAX_THREADS) {
        return NEUROCRAFT_SPMV_EINVAL;
    }
    if (n_rows == 0) {
        return NEUROCRAFT_SPMV_OK;
    }
    /* The two CSR endpoints bound every offset the row loop will read. */
    if (indptr[0] != 0 || indptr[n_rows] != nnz) {
        return NEUROCRAFT_SPMV_EDOMAIN;
    }
    if (n_threads > n_rows) {
        n_threads = (int32_t)n_rows;
    }

    single.indptr = indptr;
    single.indices = indices;
    single.values = values;
    single.x = x;
    single.y = y;
    single.nnz = nnz;
    single.n_cols = (uint64_t)n_cols;
    single.status = NEUROCRAFT_SPMV_OK;

    if (n_threads == 1) {
        single.row_begin = 0;
        single.row_end = n_rows;
        neurocraft_spmv_run_block(&single);
        return single.status;
    }

    bounds = (int64_t *)calloc((size_t)n_threads + 1u, sizeof(int64_t));
    blocks = (neurocraft_spmv_block *)calloc((size_t)n_threads, sizeof(neurocraft_spmv_block));
    threads = (neurocraft_spmv_thread *)calloc((size_t)n_threads,
                                               sizeof(neurocraft_spmv_thread));
    if (bounds == NULL || blocks == NULL || threads == NULL) {
        free(bounds);
        free(blocks);
        free(threads);
        return NEUROCRAFT_SPMV_ENOMEM;
    }

    neurocraft_spmv_partition(indptr, n_rows, nnz, n_threads, bounds);
    for (block = 0; block < n_threads; ++block) {
        blocks[block] = single;
        blocks[block].row_begin = bounds[block];
        blocks[block].row_end = bounds[block + 1];
    }

    /*
     * Blocks 1..n-1 are handed to worker threads; block 0 runs on the calling
     * thread. If native thread creation fails, the remaining blocks are executed
     * inline instead -- the row partition is unchanged, so the result stays
     * bit-identical and the call still succeeds.
     */
    for (block = 1; block < n_threads; ++block) {
        if (neurocraft_spmv_thread_start(&threads[started], &blocks[block]) != 0) {
            break;
        }
        ++started;
    }
    neurocraft_spmv_run_block(&blocks[0]);
    for (block = started + 1; block < n_threads; ++block) {
        neurocraft_spmv_run_block(&blocks[block]);
    }
    for (block = 0; block < started; ++block) {
        neurocraft_spmv_thread_join(threads[block]);
    }
    for (block = 0; block < n_threads; ++block) {
        if (blocks[block].status != NEUROCRAFT_SPMV_OK) {
            status = blocks[block].status;
        }
    }

    free(bounds);
    free(blocks);
    free(threads);
    return status;
}
