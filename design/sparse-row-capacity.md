# Sparse Row Capacity

**Status**: CPU implementation complete; CUDA validation and performance work pending

**Issue**: None yet

## Implementation Status

Last updated: 2026-05-21

Implemented:

- Added `BsrMatrix.row_ends` and `bsr_matrix_t()` support.
- Constructors and compact builders initialize compact `row_ends`.
- Active row/block lookup helpers use `row_ends` internally while preserving the public compact helper signatures.
- `BsrMatrix.uncompress_rows()` returns `-1` for slack entries.
- Topology-preserving sparse paths use active row ends: `bsr_mv`, transpose `bsr_mv`, `bsr_get_diag`, masked `bsr_axpy`, `bsr_mm` value paths, FEM Dirichlet diagonal lookup, and sparse test dense conversion helpers.
- Added `bsr_set_zero(topology="compact" | "padded" | "masked")`.
- Added graph-capture-friendly `bsr_assign(..., topology="padded", overflow="error" | "ignore")` and `bsr_copy(..., topology="padded")` paths for row-capacity-preserving copies, including block-shape-changing assignment and copy.
- Added graph-capture-friendly `bsr_set_from_triplets(..., topology="padded", overflow="error" | "ignore")` kernels that write sorted, coalesced triplets directly into existing row capacity.
- Added graph-capture-friendly `bsr_axpy(..., topology="padded", overflow="error" | "ignore")` row-merge kernels that write into existing row capacity without compact staging.
- Added graph-capture-friendly `bsr_set_transpose(..., topology="padded", overflow="error" | "ignore")` kernels that count and fill existing row capacity without compact staging.
- Added graph-capture-friendly `bsr_mm(..., topology="padded", overflow="error" | "ignore")` row scan/fill kernels that write into existing row capacity without compact staging, including aliased output cases.
- Added `status_sync()` / `status_message()` to `bsr_axpy_work_arrays` and `bsr_mm_work_arrays`; padded `bsr_axpy` and `bsr_mm` support `overflow="ignore"` with supplied work arrays to record row-capacity overflow without raising.
- Added supplied status-array support for `overflow="ignore"` on padded `bsr_set_from_triplets` and `bsr_set_transpose`.
- Added `bsr_compress(..., inplace=False)` to export a compact matrix from active entries.
- Added `bsr_compress(..., inplace=True)` as a row-local in-place sort/coalesce path for existing row capacity.
- Added post-accumulation zero pruning to `bsr_compress(..., prune_numerical_zeros=True)`.
- Added `bsr_validate()` for host-side row-capacity invariant checks.
- `bsr_validate(..., require_compact=True)` now checks compact row layout without requiring a host-synchronized exact `nnz`.
- Updated native compact transpose topology builders to read `row_ends`, so `bsr_transposed()` and compact `bsr_set_transpose()` ignore slack in gapped inputs.
- Updated masked triplet assignment to keep active topology and ignore slack through `row_ends`; native triplet mask helpers also receive `row_ends`.
- Added focused sparse tests for compact compatibility and gapped layouts.
- Added ASV benchmark coverage for gapped matvec, row compression, and padded `bsr_axpy`.
- Updated public sparse docs and API reference entries for row capacity, padded topology, `bsr_compress()`, and `bsr_validate()`.

In progress:

- CUDA validation and performance tuning for the direct padded topology-changing kernels.
- Native CPU/CUDA padded sparse routines beyond the compact transpose update.

Validated locally:

- `python3 -m compileall -q warp/_src/sparse.py warp/tests/test_sparse.py`
- `python3 -m py_compile tools/verify_sparse_row_capacity_cuda.py`
- `git diff --check`
- `PYTHONPATH=. python3 warp/tests/test_sparse.py TestSparse.test_bsr_gapped_layout_cpu`
- `PYTHONPATH=. python3 warp/tests/test_sparse.py` (`26` tests passed, `2` skipped because CUDA is not enabled in this local build)
- `PYTHONPATH=. python3 tools/verify_sparse_row_capacity_cuda.py --help`

Follow-ups:

- Investigate a tile-backed segmented sort/reduce implementation for row-local compression. The current `bsr_compress(..., inplace=True)` path is row-local and in-place, but serial within each row; a tile-backed segment sort/reduce could provide the faster production path for rows with larger candidate ranges.

Remaining implementation plan:

1. Run the CUDA verification and benchmark runbook below on a CUDA-equipped machine.
2. Replace correctness-first padded topology-changing row scans with native or tiled production implementations where ASV shows unacceptable cost.
3. Add reusable `bsr_compress` work arrays, including any candidate-to-output mapping needed by non-destructive differentiable assembly paths.
4. Wire FEM assembly to choose explicitly between destructive in-place compression and mapping-preserving assembly/compression when gradients must propagate back to original candidates.
5. Add native validation/helper routines for padded sparse matrices once the Python/kernel semantics are accepted.

Known limitations:

- Padded topology-changing ops can record overflow asynchronously with `overflow="ignore"` when status storage is supplied; `overflow="error"` still synchronizes to raise a Python exception.
- The current direct padded topology-changing kernels are correctness-first row scans; native or tiled implementations are still needed for production-scale performance.
- `bsr_compress(..., inplace=True)` is row-local but intentionally serial within each row and is not differentiability-preserving.
- Native CPU/CUDA padded sparse routines are still pending, except compact transpose topology now honors `row_ends` for gapped sources.

## CUDA Verification And Benchmark Runbook

This implementation was developed and validated locally on a CPU-only build. A CUDA-equipped reviewer should run the following checks before merge.

### 1. Build A CUDA-Enabled Warp

From the repository root:

```bash
python3 build_lib.py --quick -j 8
PYTHONPATH=. python3 -c "import warp as wp; wp.init(); assert wp.is_cuda_available(); print(wp.get_cuda_devices())"
```

If the machine does not have `libmathdx`, keep this sparse validation independent of tile MathDx setup:

```bash
python3 build_lib.py --quick --no-use-libmathdx -j 8
```

### 2. Run Sparse Correctness Tests On CUDA

Run the focused row-capacity test first:

```bash
PYTHONPATH=. python3 warp/tests/test_sparse.py TestSparse.test_bsr_gapped_layout_cuda_0
```

Run the sparse graph-capture test if the CUDA device supports Warp mempools:

```bash
PYTHONPATH=. python3 warp/tests/test_sparse.py TestSparse.test_capturability_cuda_0
```

Then run the whole sparse file:

```bash
PYTHONPATH=. python3 warp/tests/test_sparse.py
```

And run the suite runner variant before merge:

```bash
PYTHONPATH=. python3 -m warp.tests -s autodetect -p "test_sparse.py" -k TestSparse --serial-fallback
```

Expected result: all CUDA sparse tests pass. CPU-only skips are acceptable only on machines that truly lack CUDA; CUDA tests should not be skipped on the reviewer machine.

### 3. Run The Padded Graph-Capture Smoke Script

The dedicated smoke script exercises the capacity-aware graph-capture paths with `overflow="ignore"` and preallocated status/work arrays:

```bash
PYTHONPATH=. python3 tools/verify_sparse_row_capacity_cuda.py --device cuda:0
```

Expected final line:

```text
sparse row-capacity CUDA graph smoke passed on cuda:0
```

This script covers:

- `bsr_set_from_triplets(..., topology="padded", overflow="ignore")`
- `bsr_assign(..., topology="padded", overflow="ignore")`
- block-shape-changing padded assignment
- `bsr_axpy(..., topology="padded", overflow="ignore")`
- `bsr_set_transpose(..., topology="padded", overflow="ignore")`
- `bsr_mm(..., topology="padded", overflow="ignore")`
- aliased padded `bsr_mm` where `z == x`
- `bsr_copy(..., topology="padded", block_shape=(1, 1))` outside capture

The capture body must not call `status_sync()`, `.numpy()`, `nnz_sync()`, or use `overflow="error"`. Those operations are host-visible synchronization points and belong after graph replay.

For memory checking:

```bash
compute-sanitizer --tool memcheck python3 tools/verify_sparse_row_capacity_cuda.py --device cuda:0
```

Expected result: no CUDA API errors, invalid accesses, or race/memcheck failures.

### 4. Benchmark Row-Capacity Paths

Run the new focused ASV cases:

```bash
uvx --python 3.12 asv run -e --launch-method spawn -b BsrMvGappedRows HEAD^!
uvx --python 3.12 asv run -e --launch-method spawn -b BsrCompressGappedRows HEAD^!
uvx --python 3.12 asv run -e --launch-method spawn -b BsrAxpyPaddedRows HEAD^!
```

Run the existing sparse and FEM benchmarks that should not regress:

```bash
uvx --python 3.12 asv run -e --launch-method spawn -b BsrMvQuadraticTetmeshMatrix HEAD^!
uvx --python 3.12 asv run -e --launch-method spawn -b BsrMvLinearGridMatrix HEAD^!
uvx --python 3.12 asv run -e --launch-method spawn -b BsrMMQuadraticTetmeshMatrix HEAD^!
uvx --python 3.12 asv run -e --launch-method spawn -b BsrMMLinearGridMatrix HEAD^!
uvx --python 3.12 asv run -e --launch-method spawn -b FemCorotatedElasticityLinearGrid HEAD^!
```

To compare against upstream:

```bash
git fetch upstream
uvx --python 3.12 asv continuous -e --launch-method spawn -b BsrMvGappedRows upstream/main HEAD
uvx --python 3.12 asv continuous -e --launch-method spawn -b BsrCompressGappedRows upstream/main HEAD
uvx --python 3.12 asv continuous -e --launch-method spawn -b BsrAxpyPaddedRows upstream/main HEAD
uvx --python 3.12 asv continuous -e --launch-method spawn -b BsrMvQuadraticTetmeshMatrix upstream/main HEAD
uvx --python 3.12 asv continuous -e --launch-method spawn -b BsrMMQuadraticTetmeshMatrix upstream/main HEAD
uvx --python 3.12 asv continuous -e --launch-method spawn -b FemCorotatedElasticityLinearGrid upstream/main HEAD
```

Acceptance criteria:

- Existing compact sparse and FEM benchmarks should not regress outside normal ASV noise. Treat sustained regressions greater than 5 percent as blockers unless there is a documented reason.
- New row-capacity benchmarks should complete on CUDA and produce stable timings across repeated ASV runs.
- `BsrMvGappedRows` should be close to compact matvec with the same active nonzero count because slack is ignored through `row_ends`.
- `BsrCompressGappedRows` and padded topology-changing benchmarks may be slower than final native implementations; record the timings as the baseline for the correctness-first Python/kernel implementation.

## Motivation

Warp's current BSR/CSR layout stores one `offsets` array with `nrow + 1` entries. The active blocks of row `r` are assumed to live in `offsets[r]:offsets[r + 1]`, which implies that each row starts exactly where the previous row ends.

This compact layout is efficient for immutable sparse topology, but it does not support row-local spare capacity. Some simulation workloads need to reserve space for nonzeros that will be inserted later without shifting all subsequent rows. Examples include sparse matrix accumulation with changing patterns, finite element matrix assembly from row-ordered candidate entries, and in-place factorizations where fill-in is known or bounded ahead of time.

The goal is to support sparse matrices whose rows have independent active ends inside preallocated row storage, while preserving the compact layout and existing APIs for current users.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | Preserve existing compact BSR/CSR behavior by default. | Must | Existing constructors and operations should continue to produce compact matrices unless explicitly requested otherwise. |
| R2  | Support row-local spare capacity. | Must | Each row needs an active range and a capacity range. |
| R3  | Avoid host synchronization for capacity-aware fast paths. | Must | Overflow should be reported through device-side status rather than automatic synchronous fallback. |
| R4  | Keep active row lookup and block lookup efficient. | Must | Active row entries remain sorted by column. |
| R5  | Provide explicit capacity-preserving modes for topology-changing operations. | Should | Callers should opt into reusing row capacity. |
| R6  | Maintain a documented invalid-block sentinel for slack entries. | Should | `columns[b] == -1` is used for inactive slots where Warp owns or updates storage, but `row_ends` remains the source of truth. |
| R7  | Support compression of row-ordered candidate entries. | Should | This avoids global COO sorting when candidate columns and values are already partitioned by destination row, even if duplicate columns need to be accumulated. |
| R8  | Allow FEM assembly to choose a differentiable or in-place compression path. | Should | The fastest in-place compression path may overwrite the duplicate-to-output mapping needed for gradients, so differentiability should be selected at the assembly layer. |
| R9  | Avoid ASV performance benchmark regressions. | Must | Existing sparse and FEM assembly performance benchmarks should not regress; add or update ASV cases if current coverage does not exercise the changed paths. |

**Non-goals**: This design does not require changing all sparse algorithms to use gapped storage immediately. It also does not require automatic fallback from capacity reuse to compact rebuilds, because that fallback would require host synchronization when the fit decision depends on device data.

## Design

### Layout

Add a `row_ends` array to `BsrMatrix`.

```text
offsets[row]      = start of row storage
row_ends[row]     = end of active row entries
offsets[row + 1]  = end of row storage / capacity
```

The active row range is:

```text
offsets[row] : row_ends[row]
```

The row capacity range is:

```text
offsets[row] : offsets[row + 1]
```

The slack range is:

```text
row_ends[row] : offsets[row + 1]
```

The required invariant is:

```text
offsets[row] <= row_ends[row] <= offsets[row + 1]
```

Active columns must be sorted and valid:

```text
0 <= columns[b] < ncol
for b in offsets[row] : row_ends[row]
```

Slack entries are inactive. Warp-managed operations should set slack columns to `-1` where practical:

```text
columns[b] == -1
for b in row_ends[row] : offsets[row + 1]
```

The sentinel is a documented convention for consistency and debugging, but kernels must use `row_ends` to decide which blocks are active.

For compact matrices, `row_ends` has the same values as `offsets[1:]`.

### Nonzero Counts And Export

Keep `BsrMatrix.nnz` as the storage/launch upper bound. With gaps, active block indices are not dense, so `nnz` cannot simply become the active count without breaking kernels that launch over storage.

Do not add a separate active-count API in the initial design. `BsrMatrix.uncompress_rows()` may return a capacity-sized array and use `-1` for slack entries outside active row ranges.

Callers that need a compact representation should use `bsr_compress(..., inplace=False)` to produce compact `offsets`, `row_ends`, `columns`, and `values`. In that compact result, `row_ends` equals `offsets[1:]`, so the existing `nnz_sync()` behavior can recover the exact nonzero count from `offsets[nrow]` if the host needs it.

### Primitive Accessors

All sparse operations should use row access helpers rather than reading `offsets[row + 1]` as the active end.

```python
row_start = offsets[row]
row_end = row_ends[row]
row_capacity_end = offsets[row + 1]
```

`bsr_block_index()` searches only the active row range:

```python
block = lower_bound(columns, offsets[row], row_ends[row], col)
```

`bsr_row_index()` searches over `row_ends`, then rejects slack entries:

```text
row = lower_bound(row_ends, block_index + 1)
if row == nrow or block_index < offsets[row]:
    return -1
return row
```

### Capacity-Aware Operations

Topology-preserving operations only need to respect `row_ends`:

- `bsr_mv`
- transpose `bsr_mv`
- `bsr_scale`
- `bsr_get_diag`
- masked `bsr_axpy`
- masked `bsr_mm`
- `bsr_block_index`
- row decompression

Topology-changing operations keep compact behavior by default, but gain explicit padded modes:

```python
bsr_assign(src, dest, topology="padded")
bsr_set_from_triplets(dest, rows, columns, values, topology="padded", overflow="error")
bsr_compress(A, inplace=True)
bsr_axpy(x, y, topology="padded", overflow="error")
bsr_mm(x, y, z, topology="padded", overflow="error")
bsr_set_transpose(dest, src, topology="padded", overflow="error")
```

`topology="padded"` is an allocation contract: the caller promises that each destination row has enough padding for the result topology. If a row overflows, Warp records a device-side status flag and does not silently rebuild compact topology. A synchronous fallback can be added later as an explicit convenience mode, but it should not be the default capacity-aware path.

### In-Place Row Compression

Finite element assembly can produce candidate sparse entries that are already grouped by destination row, while entries within a row may contain duplicate columns from neighboring element contributions. Row capacity makes it possible to assemble these candidates directly into the destination matrix storage and then compress them in place, without rebuilding compact global CSR/BSR topology.

The proposed operation is:

```python
bsr_compress(A, prune_numerical_zeros=True, inplace=True)
```

Before the call, `A.offsets` and `A.row_ends` define row-wise candidate ranges. In this pre-compression state, `row_ends` is the end of each row's candidate entries rather than the final unique active entries:

```text
A.offsets[row] : A.row_ends[row]
```

Candidate entries must already be sorted by increasing row, meaning all entries for a given row are contiguous and live in that row's candidate range. Within a row, column indices may be unsorted and may contain duplicates. If the producer can also emit columns sorted within each row, `bsr_compress()` can use a cheaper row-local reduction path.

`bsr_compress()` sorts or otherwise coalesces entries within each row, accumulates duplicate values, optionally prunes numerical zeros, writes the unique active blocks back into the beginning of the same row range, and updates:

```text
A.row_ends[row]
```

Because compression only removes duplicates and zeros, the number of active blocks in a row cannot increase. That gives the operation stronger semantics than padded triplet construction: it can work in place when `A.columns` and `A.values` hold the candidate entries, and it does not need an overflow policy for valid input ranges. After compression, slack entries should be marked with `columns[b] == -1` where practical.

This operation is distinct from generic `bsr_set_from_triplets()`: it assumes row ordering is already provided by the producer, so it can avoid a global row/column sort and limit all topology work to independent rows. This is the path to use if FEM bilinear form assembly can emit or place candidate `(column, value)` entries in row order.

The default in-place compression path is expected to be non-differentiable with respect to candidate values. It can overwrite duplicate candidate values and discard the mapping from each original candidate entry to its compressed output block. This is likely acceptable for many assembly paths where the sparse matrix is an intermediate numerical buffer, but it should be explicit.

The differentiability choice should primarily live in FEM assembly, not in `bsr_compress()` itself. Assembly can choose whether to write row-ordered candidates directly into the matrix and call:

```python
bsr_compress(A, inplace=True)
```

or whether to use a differentiable assembly path that preserves enough information to propagate gradients from compressed values back to original element contributions.

`bsr_compress()` can still expose an `inplace` option:

```python
bsr_compress(src, dest, inplace=False, work_arrays=work)
```

where `src` contains row-ordered candidates and `dest` receives compressed values. This form avoids destructive aliasing and can preserve a candidate-to-output mapping in `work_arrays` if needed by higher-level differentiable assembly.

The important API point is that `inplace=True` should be explicit, so users and assembly code can distinguish the destructive fast path from a mapping-preserving path.

### API Argument Decisions

Use `row_ends` for the new per-row active-end array. It is concise and reads naturally in row-range code.

Use a single `topology` argument for mutually exclusive topology policies rather than adding more booleans. This unifies current `masked` behavior, current `bsr_mm(reuse_topology=True)`, and the new padded-row behavior:

```text
compact / rebuild   compute the result topology and store it compactly
masked              keep the destination active topology unchanged
cached              reuse topology data cached in work arrays from a previous call
padded              compute the result topology into existing per-row padded storage
```

The main public values are:

```python
bsr_axpy(x, y, topology="compact" | "masked" | "padded")
bsr_mm(x, y, z, topology="compact" | "masked" | "cached" | "padded")
```

`padded` contrasts with `compact`: compact output removes row padding, while padded output writes into the padding already reserved for each row.

Use the same `topology` argument for zeroing. For `bsr_set_zero()`, the policies mean:

```python
bsr_set_zero(A, topology="compact")  # zero offsets; compact empty topology
bsr_set_zero(A, topology="padded")   # keep row capacity; set row_ends[row] = offsets[row]
bsr_set_zero(A, topology="masked")   # keep active topology; zero values only
```

For copy-like operations, `topology="compact"` copies only active entries into compact storage, while `topology="padded"` preserves row padding.

For overflow handling, possible values include:

```python
overflow="error"     # record failure status if capacity is insufficient
overflow="ignore"    # leave overflowing rows undefined or unchanged, mainly for expert/internal use
overflow="fallback"  # rebuild compact topology if capacity is insufficient; requires host synchronization
```

The `"fallback"` mode should be opt-in because the fit decision depends on device data and therefore requires a host-visible branch. Another option is to avoid an `overflow` argument and always record status in the supplied work arrays:

```python
work = bsr_axpy_work_arrays()
bsr_axpy(x, y, topology="padded", work_arrays=work)
work.status_sync()
```

This keeps graph-capture-friendly behavior explicit, but it may be less discoverable for users who expect an operation to raise or report capacity errors.

Use `bsr_compress()` for row-ordered assembly compression and for producing compact representations:

```python
bsr_compress(A, prune_numerical_zeros=True, inplace=True)
compact = bsr_compress(A, prune_numerical_zeros=True, inplace=False)
```

This keeps `bsr_set_from_triplets()` focused on general COO input and gives `bsr_compress()` stronger semantics for row-ordered candidates. A possible extension is to let `bsr_compress()` accept external row-ordered candidate arrays and write into `dest`, but the core contract should remain that candidate entries are already grouped by row:

```python
bsr_compress(
    dest,
    columns=assembled_columns,
    values=assembled_values,
    row_ends=assembled_row_ends,
    inplace=False,
)
```

This extension should only be added if FEM assembly cannot conveniently write candidates directly into the matrix's own arrays.

## Staged Implementation Plan

### Stage 1: Data Model

Add `row_ends` to `BsrMatrix` and `bsr_matrix_t()`.

Update allocation helpers:

- `bsr_zeros()` creates compact empty storage with `row_ends` equal to `offsets[1:]`.
- `_bsr_resize()` refreshes `row_ends` whenever `offsets` is reallocated.
- `_bsr_ensure_fits()` remains capacity-oriented.
- Documentation clarifies that `nnz` is a storage upper bound.

### Stage 2: Row Access Primitives

Update:

- `bsr_block_index()`
- `bsr_row_index()`
- `BsrMatrix.uncompress_rows()`

This stage establishes the invariant that higher-level operations use `row_ends` for active ranges.

### Stage 3: Read-Only And Topology-Preserving Ops

Make existing operations correct for gapped matrices:

- `bsr_mv`
- transpose `bsr_mv`
- `bsr_scale`
- `bsr_get_diag`
- masked `bsr_axpy`
- masked `bsr_mm`
- dense conversion and sparse test helpers

These operations should ignore slack entries regardless of their stored value.

### Stage 4: Compact Builders Remain Compact

Keep current builders compact by default:

- `bsr_set_from_triplets`
- `bsr_from_triplets`
- `bsr_set_transpose`
- unmasked `bsr_axpy`
- unmasked `bsr_mm`

After any compact rebuild, set `row_ends` to the compact row ends.

### Stage 5: Capacity-Preserving Basics

Add capacity-preserving operations:

- `bsr_set_zero(A, topology="padded")`
- `bsr_assign(src, dest, topology="padded")`
- `bsr_copy(..., topology="padded")`

`bsr_set_zero(..., topology="padded")` sets `row_ends[row] = offsets[row]` and marks slack columns as `-1` where practical. `bsr_set_zero(..., topology="masked")` keeps active topology and zeroes values only. `bsr_set_zero(..., topology="compact")` zeroes offsets, giving a compact empty matrix.

`bsr_assign(..., topology="padded")` places each source row at `dest.offsets[row]`, updates `dest.row_ends[row]`, and leaves remaining capacity as slack.

Apply the same `topology`, `overflow`, and `inplace` naming consistently across sparse APIs.

### Stage 6: Capacity-Aware Topology Changes

Add explicit topology policies for operations that create new topology. Padded row storage is exposed as a value of the unified `topology` argument rather than as a separate boolean:

- `bsr_set_from_triplets(..., topology="padded", overflow="error")`
- `bsr_compress(..., inplace=True)`
- `bsr_axpy(..., topology="padded", overflow="error")`
- `bsr_mm(..., topology="padded", overflow="error")`
- `bsr_set_transpose(..., topology="padded", overflow="error")`

Work arrays should expose asynchronous status:

```python
work_arrays.status_sync()
```

The status path allows graph-capture-friendly execution without a host branch for fallback.

### Stage 7: Compact Representation

Initial gapped-layout support can keep `BsrMatrix.uncompress_rows()` capacity-sized and return `-1` for slack entries outside active row ranges. This is enough for callers that can filter inactive entries themselves.

When callers need a compact representation, use compression rather than a separate active-count/export API:

```python
compact = bsr_compress(A, inplace=False)
```

`bsr_compress(..., inplace=False)` returns or writes a compact matrix with no row padding. Its `row_ends` equals `offsets[1:]`, so existing compact-layout code and `nnz_sync()` can be used on the result. Callers should not assume `A.uncompress_rows()[:A.nnz]` is already compact when gaps exist.

### Stage 8: Native CPU/CUDA Support

Extend native sparse routines after the Python/kernel layout behavior is established.

Likely native additions:

- triplet build into existing row capacity
- in-place row compression/coalescing
- transpose into existing row capacity
- optional slack sentinel fill
- validation helpers

Keep existing native functions compact for compatibility.

### Stage 9: Tests And Documentation

Add tests for compact and gapped layouts:

- matvec ignores slack
- transpose matvec ignores slack
- block lookup ignores slack
- masked operations ignore slack
- `bsr_assign(..., topology="padded")` copies compact matrices into larger row capacity
- `bsr_compress(..., inplace=True)` accumulates duplicate columns in place for row-ordered candidate entries
- FEM assembly can choose a non-destructive compression/assembly path when gradients must propagate back to original candidates
- `bsr_axpy(..., topology="padded")` fills row gaps
- overflow status is set without automatic host fallback
- Warp-managed slack entries use `columns[b] == -1`

Run ASV performance benchmarks that cover sparse operations and FEM assembly. If existing benchmarks do not cover padded sparse matrices or `bsr_compress()`, add focused ASV cases before merging and check that compact-layout performance does not regress.

Update docs to explain active range, slack range, capacity range, `row_ends`, compact compression, and the invalid-block sentinel convention.

## Alternatives Considered

### Row Counts

Use a `row_counts` array and define active rows as:

```text
offsets[row] : offsets[row] + row_counts[row]
```

This is equivalent in expressive power but less convenient for compact matrices. With `row_ends`, compact matrices can use values equal to `offsets[1:]`, and row range code reads naturally.

### Full Row Address Array

Use `row_offsets` and `row_counts`, both of length `nrow`, matching MuJoCo's `rowadr` and `rownnz`.

This is clean, but it is more disruptive to the existing `BsrMatrix.offsets` API. The proposed `row_ends` design keeps the current compact `offsets` model and adds capacity semantics with minimal API disruption.

### Length-`nrow` Row Offsets

With a separate `row_ends` array, `offsets` could technically have length `nrow` instead of `nrow + 1`, with the final row capacity end inferred from `columns.shape[0]`:

```text
offsets[row]      = start of row storage
row_ends[row]     = end of active row entries
capacity_end(row) = offsets[row + 1] for row + 1 < nrow
capacity_end(row) = columns.shape[0] for row + 1 == nrow
```

This matches row-address representations more closely and removes one offset entry. It is probably more trouble than it is worth for Warp's existing API. The current code, native routines, docs, and user expectations all assume `offsets` has `nrow + 1` entries. Keeping the final sentinel also gives kernels a uniform expression for row capacity:

```text
offsets[row] : offsets[row + 1]
```

The extra integer is negligible compared with the compatibility and implementation simplicity benefits.

### Required Invalid Sentinel Only

Mark inactive slots with `columns[b] == -1` and continue using `offsets[row + 1]` as the row end.

This makes correctness depend on all slack entries being initialized and maintained, including user-provided buffers and device-side mutations. It also makes row scans pay for capacity rather than active nonzeros. The sentinel is useful as a documented convention, but `row_ends` should define correctness.

## Testing Strategy

Use `unittest`, following existing sparse tests.

Run targeted tests during development:

```bash
uv run warp/tests/test_sparse.py
```

Run the sparse test class through the suite runner before merging:

```bash
uv run --extra dev -m warp.tests -s autodetect -k TestSparse
```
