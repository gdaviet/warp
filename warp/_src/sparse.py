# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import weakref
from typing import Any, Generic, TypeVar

from numpy import eye

import warp as wp
import warp._src.utils
from warp._src.logger import log_warning
from warp._src.types import (
    Array,
    Cols,
    Rows,
    Scalar,
    Vector,
    is_array,
    scalar_types,
    type_is_matrix,
    type_repr,
    type_scalar_type,
    type_size,
    type_size_in_bytes,
    type_to_warp,
    types_equal,
)

_wp_module_name_ = "warp.sparse"

__all__ = [
    "BsrMatrix",
    "bsr_assign",
    "bsr_axpy",
    "bsr_block_index",
    "bsr_copy",
    "bsr_compress",
    "bsr_diag",
    "bsr_from_triplets",
    "bsr_get_diag",
    "bsr_identity",
    "bsr_matrix_t",
    "bsr_mm",
    "bsr_mm_work_arrays",
    "bsr_mv",
    "bsr_row_index",
    "bsr_scale",
    "bsr_set_diag",
    "bsr_set_from_triplets",
    "bsr_set_identity",
    "bsr_set_transpose",
    "bsr_set_zero",
    "bsr_transposed",
    "bsr_validate",
    "bsr_zeros",
]


# typing hints

_BlockType = TypeVar("BlockType")  # noqa: PLC0132


class _MatrixBlockType(Generic[Rows, Cols, Scalar]):
    pass


class _ScalarBlockType(Generic[Scalar]):
    pass


BlockType = _MatrixBlockType[Rows, Cols, Scalar] | _ScalarBlockType[Scalar]

_struct_cache = {}
_transfer_buffer_cache = {}

_BSR_STATUS_SUCCESS = 0
_BSR_STATUS_ROW_CAPACITY_EXCEEDED = 1


class BsrMatrix(Generic[_BlockType]):
    """Untyped base class for BSR and CSR matrices.

    Should not be constructed directly but through functions such as :func:`bsr_zeros`.

    Attributes:
        nrow (int): Number of rows of blocks.
        ncol (int): Number of columns of blocks.
        nnz (int):  Upper bound for the number of stored blocks, used for
          dimensioning launches. For compact matrices this is also the number
          of active non-zero blocks. See also :meth:`nnz_sync`.
        offsets (Array[int]): Array of size at least ``1 + nrow`` such that the
          start and capacity end indices of row ``r`` are ``offsets[r]`` and
          ``offsets[r+1]``, respectively.
        row_ends (Array[int]): Array of size at least ``nrow`` containing the
          active end index of each row. Active blocks of row ``r`` are stored in
          ``offsets[r]:row_ends[r]``. For compact matrices, ``row_ends[r]`` is
          equal to ``offsets[r+1]``.
        columns (Array[int]): Array of size at least equal to ``nnz`` containing
          block column indices.
        values (Array[BlockType]): Array of size at least equal to ``nnz``
          containing block values.
    """

    @property
    def scalar_type(self) -> Scalar:
        """Scalar type for individual block coefficients. For CSR matrices, this is the same as the block type."""
        return type_scalar_type(self.values.dtype)

    @property
    def block_shape(self) -> tuple[int, int]:
        """Shape of the individual blocks."""
        return getattr(self.values.dtype, "_shape_", (1, 1))

    @property
    def block_size(self) -> int:
        """Size of the individual blocks, i.e. number of rows per block times number of columns per block."""
        return type_size(self.values.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrix, i.e. number of rows/columns of blocks times number of rows/columns per block."""
        block_shape = self.block_shape
        return (self.nrow * block_shape[0], self.ncol * block_shape[1])

    @property
    def dtype(self) -> type:
        """Data type for individual block values."""
        return self.values.dtype

    @property
    def device(self) -> wp._src.context.Device:
        """Device on which matrix arrays are allocated."""
        return self.values.device

    @property
    def requires_grad(self) -> bool:
        """Read-only property indicating whether the matrix participates in adjoint computations."""
        return self.values.requires_grad

    @property
    def scalar_values(self) -> wp.array:
        """Access the ``values`` array as a 3d scalar array."""
        values_view = _as_3d_array(self.values, self.block_shape)
        values_view._ref = self.values  # keep ref in case we're garbage collected
        return values_view

    def uncompress_rows(self, out: wp.array = None) -> wp.array:
        """Compute the row index for each non-zero block from the compressed row offsets."""
        if out is None:
            out = wp.empty(self.nnz, dtype=int, device=self.device)

        wp.launch(
            kernel=_bsr_get_block_row,
            device=self.device,
            dim=self.nnz,
            inputs=[self.nrow, self.offsets, self.row_ends, out],
        )
        return out

    def nnz_sync(self) -> int:
        """Synchronize the number of non-zeros from the device ``offsets`` array to the host.

        Ensures that any ongoing transfer of the exact nnz number from the device offsets array to the host has completed,
        or, if none has been scheduled yet, starts a new transfer and waits for it to complete.

        Then updates the host-side nnz upper bound to match the exact one, and returns it.

        See also :meth:`notify_nnz_changed`.
        """

        buf, event = self._nnz_transfer_if_any()
        if buf is None:
            buf, event = self._copy_nnz_async()

        if event is not None:
            wp.synchronize_event(event)
        self.nnz = int(buf.numpy()[0])
        return self.nnz

    def notify_nnz_changed(self, nnz: int | None = None) -> None:
        """Notify the matrix that the number of non-zeros has been changed from outside of the :mod:`warp.sparse` builtin functions.

        Should be called in particular when the offsets array has been modified, or when the nnz upper bound has changed.
        Makes sure that the matrix is properly resized and starts the asynchronous transfer of the actual non-zero count.

        Args:
            nnz: The new upper-bound for the number of non-zeros. If not provided, it will be read from the device offsets array (requires a synchronization).
        """

        self._copy_nnz_async()
        if nnz is None:
            self.nnz_sync()

        _bsr_ensure_fits(self, nnz=nnz)

    def copy_nnz_async(self) -> None:
        """Start the asynchronous transfer of the exact nnz from the device offsets array to host and record an event for completion.

        Deprecated; prefer :meth:`notify_nnz_changed` instead, which will make sure to resize arrays if necessary.
        """
        log_warning(
            "The `copy_nnz_async` method is deprecated and will be removed in a future version. Prefer `notify_nnz_changed` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self._copy_nnz_async()

    def _copy_nnz_async(self) -> tuple[wp.array, wp.Event]:
        buf, event = self._setup_nnz_transfer()
        if buf is not None:
            stream = wp.get_stream(self.device) if self.device.is_cuda else None
            wp.copy(src=self.offsets, dest=buf, src_offset=self.nrow, count=1, stream=stream)
            if event is not None:
                stream.record_event(event, external=True)
        return buf, event

    def _setup_nnz_transfer(self) -> tuple[wp.array, wp.Event]:
        buf, event = self._nnz_transfer_if_any()
        if buf is not None:
            return buf, event

        buf, event = _allocate_transfer_buf(self.device)
        if buf is not None:
            # buf may still be None if device is currently capturing
            BsrMatrix.__setattr__(self, "_nnz_transfer", (buf, event))
            weakref.finalize(self, _redeem_transfer_buf, self.device, buf, event)

        return buf, event

    def _nnz_transfer_if_any(self) -> tuple[wp.array, wp.Event]:
        return getattr(self, "_nnz_transfer", (None, None))

    # Overloaded math operators
    def __add__(self, y):
        return bsr_axpy(y, bsr_copy(self))

    def __iadd__(self, y):
        return bsr_axpy(y, self)

    def __radd__(self, x):
        return bsr_axpy(x, bsr_copy(self))

    def __sub__(self, y):
        return bsr_axpy(y, bsr_copy(self), alpha=-1.0)

    def __rsub__(self, x):
        return bsr_axpy(x, bsr_copy(self), beta=-1.0)

    def __isub__(self, y):
        return bsr_axpy(y, self, alpha=-1.0)

    def __mul__(self, y):
        return _BsrScalingExpression(self, y)

    def __rmul__(self, x):
        return _BsrScalingExpression(self, x)

    def __imul__(self, y):
        return bsr_scale(self, y)

    def __matmul__(self, y):
        if isinstance(y, wp.array):
            return bsr_mv(self, y)

        return bsr_mm(self, y)

    def __rmatmul__(self, x):
        if isinstance(x, wp.array):
            return bsr_mv(self, x, transpose=True)

        return bsr_mm(x, self)

    def __imatmul__(self, y):
        return bsr_mm(self, y, self)

    def __truediv__(self, y):
        return _BsrScalingExpression(self, 1.0 / y)

    def __neg__(self):
        return _BsrScalingExpression(self, -1.0)

    def transpose(self):
        """Return a transposed copy of this matrix."""
        return bsr_transposed(self)


def _allocate_transfer_buf(device):
    if device.ordinal in _transfer_buffer_cache:
        all_, pool = _transfer_buffer_cache[device.ordinal]
    else:
        all_ = []
        pool = []
        _transfer_buffer_cache[device.ordinal] = (all_, pool)

    if pool:
        return pool.pop()

    if device.is_capturing:
        return None, None

    buf = wp.empty(dtype=int, shape=(1,), device="cpu", pinned=device.is_cuda)
    event = wp.Event(device) if device.is_cuda else None
    all_.append((buf, event))  # keep a reference to the buffer and event, prevent garbage collection before redeem
    return buf, event


def _redeem_transfer_buf(device, buf, event):
    _all, pool = _transfer_buffer_cache[device.ordinal]
    pool.append((buf, event))


def bsr_matrix_t(dtype: BlockType):
    dtype = type_to_warp(dtype)

    if not type_is_matrix(dtype) and dtype not in scalar_types:
        raise ValueError(f"BsrMatrix block type must be either warp matrix or scalar; got {type_repr(dtype)}")

    class BsrMatrixTyped(BsrMatrix):
        nrow: int
        """Number of rows of blocks."""
        ncol: int
        """Number of columns of blocks."""
        nnz: int
        """Upper bound for the number of non-zeros."""
        offsets: wp.array(dtype=int)
        """Array of size at least ``1 + nrow``."""
        row_ends: wp.array(dtype=int)
        """Array of size at least ``nrow``."""
        columns: wp.array(dtype=int)
        """Array of size at least equal to ``nnz``."""
        values: wp.array(dtype=dtype)

    module = wp.get_module(BsrMatrix.__module__)

    if hasattr(dtype, "_shape_"):
        type_str = f"{type_scalar_type(dtype).__name__}_{dtype._shape_[0]}_{dtype._shape_[1]}"
    else:
        type_str = dtype.__name__
    key = f"{BsrMatrix.__qualname__}_{type_str}"

    if key not in _struct_cache:
        BsrMatrixTyped.dtype = dtype  # necessary for eval_annotations
        _struct_cache[key] = wp._src.codegen.Struct(
            key=key,
            cls=BsrMatrixTyped,
            module=module,
        )

    return _struct_cache[key]


def bsr_zeros(
    rows_of_blocks: int,
    cols_of_blocks: int,
    block_type: BlockType,
    device: wp.DeviceLike = None,
) -> BsrMatrix:
    """Construct and return an empty BSR or CSR matrix with the given shape.

    Args:
        bsr: The BSR or CSR matrix to set to zero.
        rows_of_blocks: Number of rows of blocks.
        cols_of_blocks: Number of columns of blocks.
        block_type: Type of individual blocks.
          For CSR matrices, this should be a scalar type.
          For BSR matrices, this should be a matrix type (e.g. from :func:`warp.types.matrix`).
        device: Device on which to allocate the matrix arrays.
    """

    bsr = bsr_matrix_t(block_type)()

    bsr.nrow = int(rows_of_blocks)
    bsr.ncol = int(cols_of_blocks)
    bsr.nnz = 0
    bsr.columns = wp.empty(shape=(0,), dtype=int, device=device)
    bsr.values = wp.empty(shape=(0,), dtype=block_type, device=device)
    bsr.offsets = wp.zeros(shape=(bsr.nrow + 1,), dtype=int, device=device)
    bsr.row_ends = wp.zeros(shape=(bsr.nrow,), dtype=int, device=device)

    return bsr


def _bsr_resize(bsr: BsrMatrix, rows_of_blocks: int | None = None, cols_of_blocks: int | None = None) -> None:
    if rows_of_blocks is not None:
        bsr.nrow = int(rows_of_blocks)
    if cols_of_blocks is not None:
        bsr.ncol = int(cols_of_blocks)

    if bsr.offsets.size < bsr.nrow + 1:
        bsr.offsets = wp.empty(shape=(bsr.nrow + 1,), dtype=int, device=bsr.offsets.device)
    if bsr.row_ends.size < bsr.nrow:
        bsr.row_ends = wp.empty(shape=(bsr.nrow,), dtype=int, device=bsr.offsets.device)


def _bsr_set_compact_row_ends(bsr: BsrMatrix) -> None:
    if bsr.nrow > 0:
        wp.copy(dest=bsr.row_ends, src=bsr.offsets, src_offset=1, count=bsr.nrow)


def _bsr_ensure_fits(bsr: BsrMatrix, nnz: int | None = None) -> None:
    if nnz is None:
        nnz = bsr.nnz
    else:
        # update nnz upper bound
        bsr.nnz = int(nnz)

    if bsr.columns.size < nnz:
        bsr.columns = wp.empty(shape=(nnz,), dtype=int, device=bsr.columns.device)
    if bsr.values.size < nnz:
        bsr.values = wp.empty(
            shape=(nnz,), dtype=bsr.values.dtype, device=bsr.values.device, requires_grad=bsr.values.requires_grad
        )


def bsr_set_zero(
    bsr: BsrMatrix,
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
    topology: str = "compact",
):
    """Set a BSR matrix to zero, possibly changing its size.

    Args:
        bsr: The BSR or CSR matrix to set to zero.
        rows_of_blocks: If not ``None``, the new number of rows of blocks.
        cols_of_blocks: If not ``None``, the new number of columns of blocks.
        topology: Topology policy. ``"compact"`` discards the active and
          capacity topology, ``"padded"`` keeps row capacity and makes every
          row empty, and ``"masked"`` keeps active topology and zeroes values.
    """
    if topology not in ("compact", "padded", "masked"):
        raise ValueError(f"Unsupported topology policy: {topology}")

    if topology == "masked":
        if rows_of_blocks is not None or cols_of_blocks is not None:
            raise ValueError("Cannot resize a matrix with topology='masked'")
        bsr.values.zero_()
        return

    if topology == "padded" and (rows_of_blocks is not None or cols_of_blocks is not None):
        raise ValueError("Cannot resize a matrix with topology='padded'")

    _bsr_resize(bsr, rows_of_blocks, cols_of_blocks)

    if topology == "padded":
        if bsr.nrow > 0:
            wp.copy(dest=bsr.row_ends, src=bsr.offsets, count=bsr.nrow)
        if bsr.nnz > 0:
            bsr.columns.fill_(-1)
            bsr.values.zero_()
        return

    bsr.offsets.zero_()
    bsr.row_ends.zero_()
    bsr.notify_nnz_changed(nnz=0)


def _as_3d_array(arr, block_shape):
    return wp.array(
        ptr=arr.ptr,
        capacity=arr.capacity,
        device=arr.device,
        dtype=type_scalar_type(arr.dtype),
        shape=(arr.shape[0], *block_shape),
        grad=None if arr.grad is None else _as_3d_array(arr.grad, block_shape),
    )


def _optional_ctypes_pointer(array: wp.array | None, ctype):
    return None if array is None else ctypes.cast(array.ptr, ctypes.POINTER(ctype))


def _optional_ctypes_event(event: wp.Event | None):
    return None if event is None else event.cuda_event


def _bsr_status_message(status: int) -> str:
    if status == _BSR_STATUS_SUCCESS:
        return "success"
    if status == _BSR_STATUS_ROW_CAPACITY_EXCEEDED:
        return "row capacity exceeded"
    return f"unknown status {status}"


def _bsr_raise_if_status_error(status: wp.array):
    status_code = int(status.numpy()[0])
    if status_code == _BSR_STATUS_ROW_CAPACITY_EXCEEDED:
        raise RuntimeError("Destination row capacity is insufficient for topology='padded'")
    if status_code != _BSR_STATUS_SUCCESS:
        raise RuntimeError(_bsr_status_message(status_code))


def _bsr_validate_status_array(status: wp.array, device):
    if status.device != device:
        raise ValueError(f"Status and sparse matrix must reside on the same device, got {status.device} and {device}")
    if status.shape != (1,):
        raise ValueError(f"Status array must be a single-element array, got {status.shape}")
    if status.dtype != wp.int32:
        raise TypeError("Status array must be of type int32")


class _BsrStatusMixin:
    def _reset_status(self):
        self._status = None

    def _ensure_status(self, device):
        if self._status is None or self._status.device != device:
            self._status = wp.zeros(shape=(1,), dtype=int, device=device)
        else:
            self._status.zero_()

        return self._status

    def status_sync(self) -> int:
        """Return the last asynchronous sparse status code, synchronizing if needed."""

        if self._status is None:
            return _BSR_STATUS_SUCCESS

        return int(self._status.numpy()[0])

    def status_message(self) -> str:
        """Return a human-readable message for :meth:`status_sync`."""

        return _bsr_status_message(self.status_sync())


def _bsr_temp_status(device):
    return wp.zeros(shape=(1,), dtype=int, device=device)


_zero_value_masks = {
    wp.float16: 0x7FFF,
    wp.bfloat16: 0x7FFF,
    wp.float32: 0x7FFFFFFF,
    wp.float64: 0x7FFFFFFFFFFFFFFF,
    wp.int8: 0xFF,
    wp.int16: 0xFFFF,
    wp.int32: 0xFFFFFFFF,
    wp.int64: 0xFFFFFFFFFFFFFFFF,
}


def make_bsr_compress_inplace_rows(block_rows: int, block_cols: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=(block_rows, block_cols), kernel_options={"enable_backward": False})
    def bsr_compress_inplace_rows(
        prune_numerical_zeros: bool,
        offsets: wp.array(dtype=int),
        row_ends: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array3d(dtype=Any),
    ):
        row = wp.tid()

        row_beg = offsets[row]
        row_end = row_ends[row]

        # Sort the row-local candidate range by column while moving values with
        # their columns. This path is intentionally serial within each row; it
        # is a correctness-first in-place compaction path for reserved capacity.
        for block in range(row_beg + 1, row_end):
            scan = block
            while scan > row_beg and columns[scan] < columns[scan - 1]:
                col = columns[scan]
                columns[scan] = columns[scan - 1]
                columns[scan - 1] = col

                for br in range(wp.static(block_rows)):
                    for bc in range(wp.static(block_cols)):
                        value = values[scan, br, bc]
                        values[scan, br, bc] = values[scan - 1, br, bc]
                        values[scan - 1, br, bc] = value

                scan -= 1

        zero = values.dtype(0.0)
        write = row_beg
        read = row_beg

        while read < row_end:
            col = columns[read]

            if col < 0:
                read += 1
                continue

            if write != read:
                columns[write] = col
                for br in range(wp.static(block_rows)):
                    for bc in range(wp.static(block_cols)):
                        values[write, br, bc] = values[read, br, bc]

            read += 1

            while read < row_end and columns[read] == col:
                for br in range(wp.static(block_rows)):
                    for bc in range(wp.static(block_cols)):
                        values[write, br, bc] += values[read, br, bc]
                read += 1

            keep_block = True
            if prune_numerical_zeros:
                keep_block = False
                for br in range(wp.static(block_rows)):
                    for bc in range(wp.static(block_cols)):
                        if values[write, br, bc] != zero:
                            keep_block = True

            if keep_block:
                write += 1

        row_ends[row] = write

        for block in range(write, offsets[row + 1]):
            columns[block] = -1
            for br in range(wp.static(block_rows)):
                for bc in range(wp.static(block_cols)):
                    values[block, br, bc] = zero

    return bsr_compress_inplace_rows


@wp.kernel
def _bsr_accumulate_triplet_values(
    row_count: int,
    tpl_summed_offsets: wp.array(dtype=int),
    tpl_summed_indices: wp.array(dtype=int),
    tpl_values: wp.array3d(dtype=Any),
    bsr_offsets: wp.array(dtype=int),
    bsr_values: wp.array3d(dtype=Any),
):
    block, i, j = wp.tid()

    if block >= bsr_offsets[row_count]:
        return

    if block == 0:
        beg = 0
    else:
        beg = tpl_summed_offsets[block - 1]
    end = tpl_summed_offsets[block]

    val = tpl_values[tpl_summed_indices[beg], i, j]
    for k in range(beg + 1, end):
        val += tpl_values[tpl_summed_indices[k], i, j]

    bsr_values[block, i, j] = val


@wp.kernel
def _bsr_set_from_triplets_masked_values(
    count: wp.array(dtype=int),
    row_count: int,
    col_count: int,
    rows: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array3d(dtype=Any),
    bsr_offsets: wp.array(dtype=int),
    bsr_row_ends: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_values: wp.array3d(dtype=Any),
):
    triplet, i, j = wp.tid()

    if count and triplet >= count[0]:
        return

    row = rows[triplet]
    col = columns[triplet]
    if row < 0 or row >= row_count or col < 0 or col >= col_count:
        return

    block = _bsr_block_index_active(row, col, bsr_offsets, bsr_columns, bsr_row_ends)
    if block != -1:
        wp.atomic_add(bsr_values, block, i, j, values[triplet, i, j])


@wp.kernel(enable_backward=False)
def _bsr_set_from_triplets_padded_count(
    triplet_count: int,
    count: wp.array(dtype=int),
    row_count: int,
    col_count: int,
    rows: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    if row >= row_count:
        return

    active_triplet_count = triplet_count
    if count:
        active_triplet_count = wp.min(count[0], triplet_count)

    block_count = int(0)
    for triplet in range(active_triplet_count):
        tpl_row = rows[triplet]
        col = columns[triplet]
        if tpl_row != row or col < 0 or col >= col_count:
            continue

        duplicate = bool(False)
        for prev in range(triplet):
            if rows[prev] == row and columns[prev] == col:
                duplicate = True

        if not duplicate:
            block_count += 1

    row_beg = dest_offsets[row]
    capacity_end = dest_offsets[row + 1]
    if row_beg + block_count > capacity_end:
        dest_row_ends[row] = row_beg
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
        return

    row_end = row_beg + block_count
    dest_row_ends[row] = row_end

    for block in range(row_end, capacity_end):
        dest_columns[block] = -1


def make_bsr_set_from_triplets_padded_count_pruned(block_rows: int, block_cols: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=(block_rows, block_cols), kernel_options={"enable_backward": False})
    def bsr_set_from_triplets_padded_count_pruned(
        triplet_count: int,
        count: wp.array(dtype=int),
        row_count: int,
        col_count: int,
        rows: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array3d(dtype=Any),
        dest_offsets: wp.array(dtype=int),
        dest_row_ends: wp.array(dtype=int),
        dest_columns: wp.array(dtype=int),
        status: wp.array(dtype=int),
    ):
        row = wp.tid()

        if row >= row_count:
            return

        active_triplet_count = triplet_count
        if count:
            active_triplet_count = wp.min(count[0], triplet_count)

        zero = values.dtype(0.0)
        block_count = int(0)
        for triplet in range(active_triplet_count):
            tpl_row = rows[triplet]
            col = columns[triplet]
            if tpl_row != row or col < 0 or col >= col_count:
                continue

            duplicate = bool(False)
            for prev in range(triplet):
                if rows[prev] == row and columns[prev] == col:
                    duplicate = True

            if not duplicate:
                keep_block = bool(False)
                for br in range(wp.static(block_rows)):
                    for bc in range(wp.static(block_cols)):
                        value = zero
                        for dup in range(triplet, active_triplet_count):
                            if rows[dup] == row and columns[dup] == col:
                                value += values[dup, br, bc]

                        if value != zero:
                            keep_block = True

                if keep_block:
                    block_count += 1

        row_beg = dest_offsets[row]
        capacity_end = dest_offsets[row + 1]
        if row_beg + block_count > capacity_end:
            dest_row_ends[row] = row_beg
            wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
            return

        row_end = row_beg + block_count
        dest_row_ends[row] = row_end

        for block in range(row_end, capacity_end):
            dest_columns[block] = -1

    return bsr_set_from_triplets_padded_count_pruned


@wp.kernel(enable_backward=False)
def _bsr_set_from_triplets_padded_fill_columns(
    triplet_count: int,
    count: wp.array(dtype=int),
    row_count: int,
    col_count: int,
    rows: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
):
    row = wp.tid()

    if row >= row_count:
        return

    active_triplet_count = triplet_count
    if count:
        active_triplet_count = wp.min(count[0], triplet_count)

    previous_col = int(-1)
    for block in range(dest_offsets[row], dest_row_ends[row]):
        next_col = col_count
        for triplet in range(active_triplet_count):
            col = columns[triplet]
            if rows[triplet] == row and col > previous_col and col < next_col and col >= 0 and col < col_count:
                next_col = col

        dest_columns[block] = next_col
        previous_col = next_col


def make_bsr_set_from_triplets_padded_fill_values(block_rows: int, block_cols: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=(block_rows, block_cols), kernel_options={"enable_backward": False})
    def bsr_set_from_triplets_padded_fill_values(
        prune_numerical_zeros: bool,
        triplet_count: int,
        count: wp.array(dtype=int),
        row_count: int,
        col_count: int,
        rows: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array3d(dtype=Any),
        dest_offsets: wp.array(dtype=int),
        dest_row_ends: wp.array(dtype=int),
        dest_columns: wp.array(dtype=int),
        dest_values: wp.array3d(dtype=Any),
    ):
        row, br, bc = wp.tid()

        if row >= row_count:
            return

        active_triplet_count = triplet_count
        if count:
            active_triplet_count = wp.min(count[0], triplet_count)

        zero = values.dtype(0.0)
        previous_col = int(-1)
        for block in range(dest_offsets[row], dest_row_ends[row]):
            next_col = col_count
            for triplet in range(active_triplet_count):
                col = columns[triplet]
                if rows[triplet] == row and col > previous_col and col < next_col and col >= 0 and col < col_count:
                    keep_block = bool(True)
                    if prune_numerical_zeros:
                        keep_block = bool(False)
                        for keep_br in range(wp.static(block_rows)):
                            for keep_bc in range(wp.static(block_cols)):
                                keep_value = zero
                                for dup in range(active_triplet_count):
                                    if rows[dup] == row and columns[dup] == col:
                                        keep_value += values[dup, keep_br, keep_bc]

                                if keep_value != zero:
                                    keep_block = True

                    if keep_block:
                        next_col = col

            value = zero
            for triplet in range(active_triplet_count):
                if rows[triplet] == row and columns[triplet] == next_col:
                    value += values[triplet, br, bc]

            if br == 0 and bc == 0:
                dest_columns[block] = next_col
            dest_values[block, br, bc] = value
            previous_col = next_col

    return bsr_set_from_triplets_padded_fill_values


def bsr_set_from_triplets(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    rows: Array[int],
    columns: Array[int],
    values: Array[Scalar | BlockType[Rows, Cols, Scalar]] | None = None,
    count: Array[int] | None = None,
    prune_numerical_zeros: bool = True,
    masked: bool = False,
    topology: str | None = None,
    overflow: str = "error",
    status: wp.array | None = None,
):
    """Fill a BSR matrix with values defined by coordinate-oriented (COO) triplets, discarding existing blocks.

    The first dimension of the three input arrays must match and indicates the number of COO triplets.

    Args:
        dest: Sparse matrix to populate.
        rows: Row index for each non-zero.
        columns: Columns index for each non-zero.
        values: Block values for each non-zero. Must be either a one-dimensional array with data type identical
          to the ``dest`` matrix's block type, or a 3d array with data type equal to the ``dest`` matrix's scalar type.
          If ``None``, the values array of the resulting matrix will be allocated but uninitialized.
        count: Single-element array indicating the number of triplets. If ``None``, the number of triplets is determined from the shape of
          ``rows`` and ``columns`` arrays.
        prune_numerical_zeros: If ``True``, will ignore the zero-valued blocks.
        masked: If ``True``, ignore blocks that are not existing non-zeros of ``dest``.
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          ``masked=True``, and ``"padded"`` writes the compacted triplet
          topology into existing destination row capacity.
        overflow: Overflow policy for ``topology="padded"``. ``"error"``
          raises on insufficient row capacity, while ``"ignore"`` records
          status in ``status`` and leaves overflowing rows undefined.
        status: Optional single-element int array receiving asynchronous status
          for ``topology="padded"``. Required when ``overflow="ignore"``.
    """
    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if overflow not in ("error", "ignore"):
        raise NotImplementedError("Only overflow='error' and overflow='ignore' are currently implemented")
    if status is not None:
        _bsr_validate_status_array(status, dest.device)
        status.zero_()

    masked = topology == "masked"

    if rows.device != columns.device or rows.device != dest.device:
        raise ValueError(
            f"Rows and columns must reside on the destination matrix device, got {rows.device}, {columns.device} and {dest.device}"
        )

    if rows.shape[0] != columns.shape[0]:
        raise ValueError(
            f"Rows and columns arrays must have the same length, got {rows.shape[0]} and {columns.shape[0]}"
        )

    if rows.dtype != wp.int32 or columns.dtype != wp.int32:
        raise TypeError("Rows and columns arrays must be of type int32")

    if count is not None:
        if count.device != rows.device:
            raise ValueError(f"Count and rows must reside on the same device, got {count.device} and {rows.device}")

        if count.shape != (1,):
            raise ValueError(f"Count array must be a single-element array, got {count.shape}")

        if count.dtype != wp.int32:
            raise TypeError("Count array must be of type int32")

    # Accept either array1d(dtype) or contiguous array3d(scalar_type) as values
    if values is not None:
        if values.device != rows.device:
            raise ValueError(f"Values and rows must reside on the same device, got {values.device} and {rows.device}")

        if values.shape[0] != rows.shape[0]:
            raise ValueError(
                f"Values and rows arrays must have the same length, got {values.shape[0]} and {rows.shape[0]}"
            )

        if values.ndim == 1:
            if not types_equal(values.dtype, dest.values.dtype):
                raise ValueError(
                    f"Values array type must correspond to that of the dest matrix, got {type_repr(values.dtype)} and {type_repr(dest.values.dtype)}"
                )
        elif values.ndim == 3:
            if values.shape[1:] != dest.block_shape:
                raise ValueError(
                    f"Last two dimensions in values array ({values.shape[1:]}) should correspond to matrix block shape {(dest.block_shape)})"
                )

            if type_scalar_type(values.dtype) != dest.scalar_type:
                raise ValueError(
                    f"Scalar type of values array ({type_repr(values.dtype)}) should correspond to that of matrix ({type_repr(dest.scalar_type)})"
                )
        else:
            raise ValueError(f"Number of dimension for values array should be 1 or 3, got {values.ndim}")

        if prune_numerical_zeros and not values.is_contiguous:
            raise ValueError("Values array should be contiguous for numerical zero pruning")

    nnz = rows.shape[0]
    if nnz == 0:
        bsr_set_zero(dest, topology="padded" if topology == "padded" else "compact")
        return

    if topology == "padded":
        if overflow == "ignore" and status is None:
            raise ValueError("`status` must be supplied when using overflow='ignore'")

        check_status = overflow == "error"
        if status is None:
            status = _bsr_temp_status(dest.device)

        if values is not None and prune_numerical_zeros:
            wp.launch(
                make_bsr_set_from_triplets_padded_count_pruned(*dest.block_shape),
                dim=dest.nrow,
                device=dest.device,
                inputs=[
                    nnz,
                    count,
                    dest.nrow,
                    dest.ncol,
                    rows,
                    columns,
                    _as_3d_array(values, dest.block_shape),
                    dest.offsets,
                    dest.row_ends,
                    dest.columns,
                    status,
                ],
            )
        else:
            wp.launch(
                _bsr_set_from_triplets_padded_count,
                dim=dest.nrow,
                device=dest.device,
                inputs=[
                    nnz,
                    count,
                    dest.nrow,
                    dest.ncol,
                    rows,
                    columns,
                    dest.offsets,
                    dest.row_ends,
                    dest.columns,
                    status,
                ],
            )

        if check_status:
            _bsr_raise_if_status_error(status)

        if values is None:
            wp.launch(
                _bsr_set_from_triplets_padded_fill_columns,
                dim=dest.nrow,
                device=dest.device,
                inputs=[
                    nnz,
                    count,
                    dest.nrow,
                    dest.ncol,
                    rows,
                    columns,
                    dest.offsets,
                    dest.row_ends,
                    dest.columns,
                ],
            )
        else:
            wp.launch(
                make_bsr_set_from_triplets_padded_fill_values(*dest.block_shape),
                dim=(dest.nrow, *dest.block_shape),
                device=dest.device,
                inputs=[
                    prune_numerical_zeros,
                    nnz,
                    count,
                    dest.nrow,
                    dest.ncol,
                    rows,
                    columns,
                    _as_3d_array(values, dest.block_shape),
                    dest.offsets,
                    dest.row_ends,
                    dest.columns,
                    dest.scalar_values,
                ],
            )
        return

    if masked:
        dest.values.zero_()
        if values is not None:
            wp.launch(
                _bsr_set_from_triplets_masked_values,
                dim=(nnz, *dest.block_shape),
                device=dest.device,
                inputs=[
                    count,
                    dest.nrow,
                    dest.ncol,
                    rows,
                    columns,
                    _as_3d_array(values, dest.block_shape),
                    dest.offsets,
                    dest.row_ends,
                    dest.columns,
                    dest.scalar_values,
                ],
            )
        return

    # Increase dest array sizes if needed
    _bsr_ensure_fits(dest, nnz=nnz)

    device = dest.values.device
    scalar_type = dest.scalar_type
    zero_value_mask = _zero_value_masks.get(scalar_type, 0) if prune_numerical_zeros else 0

    # compute the BSR topology

    from warp._src.context import runtime  # noqa: PLC0415

    if device.is_cpu:
        native_func = runtime.core.wp_bsr_matrix_from_triplets_host
    else:
        native_func = runtime.core.wp_bsr_matrix_from_triplets_device

    nnz_buf, nnz_event = dest._setup_nnz_transfer()
    summed_triplet_offsets = wp.empty(shape=(nnz,), dtype=wp.int32, device=device)
    summed_triplet_indices = wp.empty(shape=(nnz,), dtype=wp.int32, device=device)

    with wp.ScopedDevice(device):
        native_func(
            dest.block_size,
            type_size_in_bytes(scalar_type),
            dest.nrow,
            dest.ncol,
            nnz,
            _optional_ctypes_pointer(count, ctype=ctypes.c_int32),
            ctypes.cast(rows.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(values, ctype=ctypes.c_int32),
            zero_value_mask,
            masked,
            ctypes.cast(summed_triplet_offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(summed_triplet_indices.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(dest.row_ends.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
            _optional_ctypes_event(nnz_event),
        )

        # now accumulate repeated blocks
        wp.launch(
            _bsr_accumulate_triplet_values,
            dim=(nnz, *dest.block_shape),
            inputs=[
                dest.nrow,
                summed_triplet_offsets,
                summed_triplet_indices,
                _as_3d_array(values, dest.block_shape),
                dest.offsets,
            ],
            outputs=[dest.scalar_values],
        )

        if not masked:
            _bsr_set_compact_row_ends(dest)


def bsr_from_triplets(
    rows_of_blocks: int,
    cols_of_blocks: int,
    rows: Array[int],
    columns: Array[int],
    values: Array[Scalar | BlockType[Rows, Cols, Scalar]],
    prune_numerical_zeros: bool = True,
):
    """Construct a BSR matrix with values defined by coordinate-oriented (COO) triplets.

    The first dimension of the three input arrays must match and indicates the number of COO triplets.

    Args:
        rows_of_blocks: Number of rows of blocks.
        cols_of_blocks: Number of columns of blocks.
        rows: Row index for each non-zero.
        columns: Columns index for each non-zero.
        values: Block values for each non-zero. Must be either a one-dimensional array with data type identical
          to the ``dest`` matrix's block type, or a 3d array with data type equal to the ``dest`` matrix's scalar type.
        prune_numerical_zeros: If ``True``, will ignore the zero-valued blocks.
    """

    if values.ndim == 3:
        block_type = wp.types.matrix(shape=values.shape[1:], dtype=values.dtype)
    else:
        block_type = values.dtype

    A = bsr_zeros(
        rows_of_blocks=rows_of_blocks, cols_of_blocks=cols_of_blocks, block_type=block_type, device=values.device
    )
    A.values.requires_grad = values.requires_grad
    bsr_set_from_triplets(A, rows, columns, values, prune_numerical_zeros=prune_numerical_zeros)
    return A


def bsr_compress(
    src: BsrMatrixOrExpression[BlockType[Rows, Cols, Scalar]],
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]] | None = None,
    prune_numerical_zeros: bool = True,
    inplace: bool = False,
    work_arrays=None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """Compress the active blocks of ``src``.

    Slack entries outside ``offsets[row]:row_ends[row]`` are ignored. When
    ``inplace=False``, duplicate active entries are accumulated by the same
    compact COO builder used by :func:`bsr_set_from_triplets` and ``dest`` is a
    compact matrix. When ``inplace=True``, entries are sorted and coalesced
    independently within each active source row.

    Args:
        src: Matrix to compact.
        dest: Optional destination matrix. If omitted, a new compact matrix is
          allocated.
        prune_numerical_zeros: If ``True``, zero-valued blocks are pruned.
        inplace: If ``True``, sort and coalesce each active row range directly
          in ``src`` using its existing row capacity.
        work_arrays: Reserved for future in-place and differentiable paths.
    """

    if work_arrays is not None:
        raise NotImplementedError("bsr_compress work arrays are not implemented yet")

    if inplace:
        if dest is not None:
            raise ValueError("Cannot provide 'dest' when bsr_compress(..., inplace=True)")
        if not isinstance(src, BsrMatrix):
            raise ValueError("bsr_compress(..., inplace=True) requires a concrete BsrMatrix")

        if src.nrow > 0:
            wp.launch(
                kernel=make_bsr_compress_inplace_rows(*src.block_shape),
                device=src.device,
                dim=src.nrow,
                inputs=[
                    prune_numerical_zeros,
                    src.offsets,
                    src.row_ends,
                    src.columns,
                    src.scalar_values,
                ],
            )
        return src

    src, scale = _extract_matrix_and_scale(src)

    if dest is None:
        dest = bsr_zeros(src.nrow, src.ncol, block_type=src.values.dtype, device=src.device)
        dest.values.requires_grad = src.requires_grad
    else:
        if dest.device != src.device:
            raise ValueError(
                f"Source and destination matrices must reside on the same device, got {src.device} and {dest.device}"
            )
        if dest.block_shape != src.block_shape or dest.scalar_type != src.scalar_type:
            raise ValueError(
                "Source and destination matrices must have the same block type, got "
                f"({src.block_shape}, {src.scalar_type}) and ({dest.block_shape}, {dest.scalar_type})"
            )
        _bsr_resize(dest, rows_of_blocks=src.nrow, cols_of_blocks=src.ncol)

    rows = src.uncompress_rows()
    bsr_set_from_triplets(
        dest,
        rows,
        src.columns[: src.nnz],
        src.values[: src.nnz],
        prune_numerical_zeros=prune_numerical_zeros,
    )

    if scale != 1.0:
        bsr_scale(dest, scale)

    if prune_numerical_zeros and dest.nnz > 0:
        pruned = bsr_zeros(dest.nrow, dest.ncol, block_type=dest.values.dtype, device=dest.device)
        pruned.values.requires_grad = dest.requires_grad
        bsr_set_from_triplets(
            pruned,
            dest.uncompress_rows(),
            dest.columns[: dest.nnz],
            dest.values[: dest.nnz],
            prune_numerical_zeros=True,
        )
        bsr_assign(dest=dest, src=pruned)

    return dest


class _BsrExpression(Generic[_BlockType]):
    pass


class _BsrScalingExpression(_BsrExpression):
    def __init__(self, mat, scale):
        self.mat = mat
        self.scale = scale

    def eval(self):
        return bsr_copy(self)

    @property
    def nrow(self) -> int:
        return self.mat.nrow

    @property
    def ncol(self) -> int:
        return self.mat.ncol

    @property
    def nnz(self) -> int:
        return self.mat.nnz

    @property
    def offsets(self) -> wp.array:
        return self.mat.offsets

    @property
    def row_ends(self) -> wp.array:
        return self.mat.row_ends

    @property
    def columns(self) -> wp.array:
        return self.mat.columns

    @property
    def scalar_type(self) -> Scalar:
        return self.mat.scalar_type

    @property
    def block_shape(self) -> tuple[int, int]:
        return self.mat.block_shape

    @property
    def block_size(self) -> int:
        return self.mat.block_size

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    @property
    def dtype(self) -> type:
        return self.mat.dtype

    @property
    def requires_grad(self) -> bool:
        return self.mat.requires_grad

    @property
    def device(self) -> wp._src.context.Device:
        return self.mat.device

    # Overloaded math operators
    def __add__(self, y):
        return bsr_axpy(y, bsr_copy(self.mat), alpha=self.scale)

    def __radd__(self, x):
        return bsr_axpy(x, bsr_copy(self.mat), beta=self.scale)

    def __sub__(self, y):
        return bsr_axpy(y, bsr_copy(self.mat), alpha=-self.scale)

    def __rsub__(self, x):
        return bsr_axpy(x, bsr_copy(self.mat), beta=-self.scale)

    def __mul__(self, y):
        return _BsrScalingExpression(self.mat, y * self.scale)

    def __rmul__(self, x):
        return _BsrScalingExpression(self.mat, x * self.scale)

    def __matmul__(self, y):
        if isinstance(y, wp.array):
            return bsr_mv(self.mat, y, alpha=self.scale)

        return bsr_mm(self.mat, y, alpha=self.scale)

    def __rmatmul__(self, x):
        if isinstance(x, wp.array):
            return bsr_mv(self.mat, x, alpha=self.scale, transpose=True)

        return bsr_mm(x, self.mat, alpha=self.scale)

    def __truediv__(self, y):
        return _BsrScalingExpression(self.mat, self.scale / y)

    def __neg__(self):
        return _BsrScalingExpression(self.mat, -self.scale)

    def transpose(self):
        """Return a transposed copy of this matrix."""
        return _BsrScalingExpression(self.mat.transpose(), self.scale)


BsrMatrixOrExpression = BsrMatrix[_BlockType] | _BsrExpression[_BlockType]


def _extract_matrix_and_scale(bsr: BsrMatrixOrExpression):
    if isinstance(bsr, BsrMatrix):
        return bsr, 1.0
    if isinstance(bsr, _BsrScalingExpression):
        return bsr.mat, bsr.scale

    raise ValueError("Argument cannot be interpreted as a BsrMatrix")


def bsr_validate(
    A: BsrMatrixOrExpression,
    require_sorted: bool = True,
    require_unique: bool = True,
    require_slack_sentinel: bool = False,
    require_compact: bool = False,
    raise_on_error: bool = True,
) -> bool:
    """Validate BSR/CSR row-capacity invariants on the host.

    This helper synchronizes matrix metadata to the host. It is intended for
    tests, debugging, and defensive validation around user-provided sparse
    buffers.

    Args:
        A: Matrix to validate.
        require_sorted: If ``True``, require active column indices to be sorted
          within each row.
        require_unique: If ``True``, require active column indices to be unique
          within each row.
        require_slack_sentinel: If ``True``, require every slack slot in
          ``row_ends[row]:offsets[row + 1]`` to have column ``-1``.
        require_compact: If ``True``, require ``row_ends`` to equal
          ``offsets[1:]``. The host-side ``A.nnz`` value may still be an upper
          bound until :meth:`BsrMatrix.nnz_sync` is called.
        raise_on_error: If ``True``, raise ``ValueError`` on the first
          validation failure. Otherwise return ``False``.
    """

    A, _ = _extract_matrix_and_scale(A)

    def fail(message: str) -> bool:
        if raise_on_error:
            raise ValueError(message)
        return False

    if A.offsets.size < A.nrow + 1:
        return fail(f"offsets array must have at least {A.nrow + 1} entries, got {A.offsets.size}")
    if A.row_ends.size < A.nrow:
        return fail(f"row_ends array must have at least {A.nrow} entries, got {A.row_ends.size}")
    if A.columns.shape[0] < A.nnz:
        return fail(f"columns array must have at least A.nnz entries, got {A.columns.shape[0]} and {A.nnz}")
    if A.values.shape[0] < A.nnz:
        return fail(f"values array must have at least A.nnz entries, got {A.values.shape[0]} and {A.nnz}")

    offsets = A.offsets.numpy()[: A.nrow + 1]
    row_ends = A.row_ends.numpy()[: A.nrow]
    columns = A.columns.numpy()[: A.nnz]

    if int(offsets[0]) != 0:
        return fail(f"offsets[0] must be 0, got {int(offsets[0])}")

    for row in range(A.nrow):
        row_beg = int(offsets[row])
        capacity_end = int(offsets[row + 1])
        row_end = int(row_ends[row])

        if row_beg > capacity_end:
            return fail(f"offsets must be nondecreasing, got offsets[{row}] > offsets[{row + 1}]")
        if row_beg < 0:
            return fail(f"offsets[{row}] must be nonnegative, got {row_beg}")
        if row_end < row_beg or row_end > capacity_end:
            return fail(
                f"row_ends[{row}] must satisfy offsets[{row}] <= row_ends[{row}] <= offsets[{row + 1}]"
            )
        if capacity_end > A.nnz:
            return fail(f"offsets[{row + 1}] must be no larger than A.nnz, got {capacity_end} and {A.nnz}")
        if require_compact and row_end != capacity_end:
            return fail(f"row_ends[{row}] must equal offsets[{row + 1}] for compact matrices")

        previous_col = -1
        seen_cols = set()
        for block in range(row_beg, row_end):
            col = int(columns[block])
            if col < 0 or col >= A.ncol:
                return fail(f"active column at block {block} must be in [0, {A.ncol}), got {col}")
            if require_sorted and block > row_beg and col < previous_col:
                return fail(f"active columns must be sorted within row {row}")
            if require_unique:
                if col in seen_cols:
                    return fail(f"active columns must be unique within row {row}, duplicate column {col}")
                seen_cols.add(col)
            previous_col = col

        if require_slack_sentinel:
            for block in range(row_end, capacity_end):
                if int(columns[block]) != -1:
                    return fail(f"slack column at block {block} must be -1, got {int(columns[block])}")

    return True


@wp.func
def bsr_row_index(
    offsets: wp.array(dtype=int),
    row_count: int,
    block_index: int,
) -> int:
    """Return the index of the row containing a given block, or -1 if no such row exists.

    Args:
        offsets: Array of size at least ``1 + row_count`` containing the offsets of the blocks in each row.
        row_count: Number of rows of blocks.
        block_index: Index of the block.
    """
    return wp.where(block_index < offsets[row_count], wp.lower_bound(offsets, 0, row_count + 1, block_index + 1), 0) - 1


@wp.func
def _bsr_row_index_active(
    offsets: wp.array(dtype=int),
    row_count: int,
    block_index: int,
    row_ends: wp.array(dtype=int),
) -> int:
    """Return the row containing an active block in a capacity-aware BSR matrix."""

    row = wp.lower_bound(row_ends, 0, row_count, block_index + 1)
    if row == row_count:
        return -1
    if block_index < offsets[row]:
        return -1
    if block_index >= row_ends[row]:
        return -1
    return row


@wp.func
def bsr_block_index(
    row: int,
    col: int,
    bsr_offsets: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
) -> int:
    """Return the index of the block at block-coordinates (row, col), or -1 if no such block exists.

    Assumes that the segments of ``bsr_columns`` corresponding to each row are sorted.

    Args:
        row: Row of the block.
        col: Column of the block.
        bsr_offsets: Array of size at least ``1 + row`` containing the offsets of the blocks in each row.
        bsr_columns: Array of size at least equal to ``bsr_offsets[row + 1]`` containing the column indices of the blocks.
    """

    if row < 0:
        return -1

    row_beg = bsr_offsets[row]
    row_end = bsr_offsets[row + 1]

    if row_beg == row_end:
        return -1

    block_index = wp.lower_bound(bsr_columns, row_beg, row_end, col)
    if block_index == row_end:
        return -1
    return wp.where(bsr_columns[block_index] == col, block_index, -1)


@wp.func
def _bsr_block_index_active(
    row: int,
    col: int,
    bsr_offsets: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_row_ends: wp.array(dtype=int),
) -> int:
    """Return the active block index in a capacity-aware BSR matrix."""

    if row < 0:
        return -1

    row_beg = bsr_offsets[row]
    row_end = bsr_row_ends[row]

    if row_beg == row_end:
        return -1

    block_index = wp.lower_bound(bsr_columns, row_beg, row_end, col)
    if block_index == row_end:
        return -1
    return wp.where(bsr_columns[block_index] == col, block_index, -1)


@wp.kernel(enable_backward=False)
def _bsr_assign_list_blocks(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dest_rows: wp.array(dtype=int),
    dest_cols: wp.array(dtype=int),
):
    block, subrow, subcol = wp.tid()
    dest_block = (block * src_subcols + subcol) * src_subrows + subrow

    row = _bsr_row_index_active(src_offsets, src_row_count, block, src_row_ends)
    if row == -1:
        dest_rows[dest_block] = row  # invalid
        dest_cols[dest_block] = row
    else:
        dest_subrow = row * src_subrows + subrow
        dest_subcol = src_columns[block] * src_subcols + subcol
        dest_rows[dest_block] = dest_subrow // dest_subrows
        dest_cols[dest_block] = dest_subcol // dest_subcols


@wp.kernel
def _bsr_assign_copy_blocks(
    scale: Any,
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    src_block = wp.tid()
    src_block, subrow, subcol = wp.tid()

    src_row = _bsr_row_index_active(src_offsets, src_row_count, src_block, src_row_ends)
    if src_row == -1:
        return

    src_col = src_columns[src_block]

    dest_subrow = src_row * src_subrows + subrow
    dest_subcol = src_col * src_subcols + subcol
    dest_row = dest_subrow // dest_subrows
    dest_col = dest_subcol // dest_subcols

    dest_block = _bsr_block_index_active(dest_row, dest_col, dest_offsets, dest_columns, dest_row_ends)
    if dest_block == -1:
        return

    split_row = dest_subrow - dest_subrows * dest_row
    split_col = dest_subcol - dest_subcols * dest_col

    rows_per_subblock = src_values.shape[1] // src_subrows
    cols_per_subblock = src_values.shape[2] // src_subcols

    dest_base_i = split_row * rows_per_subblock
    dest_base_j = split_col * cols_per_subblock

    src_base_i = subrow * rows_per_subblock
    src_base_j = subcol * cols_per_subblock

    for i in range(rows_per_subblock):
        for j in range(cols_per_subblock):
            dest_values[dest_block, i + dest_base_i, j + dest_base_j] = dest_values.dtype(
                scale * src_values[src_block, i + src_base_i, j + src_base_j]
            )


@wp.kernel(enable_backward=False)
def _bsr_assign_padded_row_ranges(
    row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    if row >= row_count:
        return

    src_beg = src_offsets[row]
    src_end = src_row_ends[row]
    src_count = src_end - src_beg

    dest_beg = dest_offsets[row]
    dest_capacity_end = dest_offsets[row + 1]
    dest_count = dest_capacity_end - dest_beg

    if src_count > dest_count:
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
        return

    dest_active_end = dest_beg + src_count
    dest_row_ends[row] = dest_active_end

    for block in range(dest_active_end, dest_capacity_end):
        dest_columns[block] = -1


@wp.func
def _bsr_ranges_overlap(first_a: int, count_a: int, first_b: int, count_b: int) -> bool:
    return first_a < first_b + count_b and first_b < first_a + count_a


@wp.kernel(enable_backward=False)
def _bsr_assign_padded_reblock_topology(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    dest_row_count: int,
    dest_col_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    dest_row = wp.tid()

    if dest_row >= dest_row_count:
        return

    block_count = int(0)
    dest_subrow_first = dest_row * dest_subrows

    for dest_col in range(dest_col_count):
        dest_subcol_first = dest_col * dest_subcols
        found = bool(False)

        for src_row in range(src_row_count):
            src_subrow_first = src_row * src_subrows
            if _bsr_ranges_overlap(src_subrow_first, src_subrows, dest_subrow_first, dest_subrows):
                for src_block in range(src_offsets[src_row], src_row_ends[src_row]):
                    src_subcol_first = src_columns[src_block] * src_subcols
                    if _bsr_ranges_overlap(src_subcol_first, src_subcols, dest_subcol_first, dest_subcols):
                        found = True

        if found:
            block_count += 1

    dest_beg = dest_offsets[dest_row]
    capacity_end = dest_offsets[dest_row + 1]
    row_end = dest_beg + block_count

    if row_end > capacity_end:
        dest_row_ends[dest_row] = dest_beg
        for block in range(dest_beg, capacity_end):
            dest_columns[block] = -1
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
        return

    dest_row_ends[dest_row] = row_end

    dest_block = dest_beg
    for dest_col in range(dest_col_count):
        dest_subcol_first = dest_col * dest_subcols
        found = bool(False)

        for src_row in range(src_row_count):
            src_subrow_first = src_row * src_subrows
            if _bsr_ranges_overlap(src_subrow_first, src_subrows, dest_subrow_first, dest_subrows):
                for src_block in range(src_offsets[src_row], src_row_ends[src_row]):
                    src_subcol_first = src_columns[src_block] * src_subcols
                    if _bsr_ranges_overlap(src_subcol_first, src_subcols, dest_subcol_first, dest_subcols):
                        found = True

        if found:
            dest_columns[dest_block] = dest_col
            dest_block += 1

    for block in range(row_end, capacity_end):
        dest_columns[block] = -1


@wp.kernel(enable_backward=False)
def _bsr_copy_reblocked_capacity_counts(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    src_row_count: int,
    dest_row_count: int,
    src_offsets: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
):
    dest_row = wp.tid()

    if dest_row >= dest_row_count:
        return

    if dest_row == 0:
        dest_offsets[0] = 0

    dest_subrow_first = dest_row * dest_subrows
    block_count = int(0)

    for src_row in range(src_row_count):
        src_subrow_first = src_row * src_subrows
        if _bsr_ranges_overlap(src_subrow_first, src_subrows, dest_subrow_first, dest_subrows):
            block_count += (src_offsets[src_row + 1] - src_offsets[src_row]) * src_subcols

    dest_offsets[dest_row + 1] = block_count


@wp.kernel
def _bsr_assign_padded_copy_blocks(
    scale: Any,
    structure_only: bool,
    row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    src_block, br, bc = wp.tid()

    src_row = _bsr_row_index_active(src_offsets, row_count, src_block, src_row_ends)
    if src_row == -1:
        return

    dest_block = dest_offsets[src_row] + src_block - src_offsets[src_row]
    if dest_block >= dest_row_ends[src_row]:
        return

    if br == 0 and bc == 0:
        dest_columns[dest_block] = src_columns[src_block]

    if not structure_only:
        dest_values[dest_block, br, bc] = dest_values.dtype(scale * src_values[src_block, br, bc])


def _bsr_assign_padded_same_block(
    dest: BsrMatrix,
    src: BsrMatrix,
    src_scale: float = 1.0,
    structure_only: bool = False,
    status: wp.array | None = None,
    check_status: bool = True,
):
    if dest.block_shape != src.block_shape:
        raise ValueError("Padded same-block assignment requires matching block shapes")

    if status is None:
        status = _bsr_temp_status(dest.device)
    else:
        status.zero_()

    wp.launch(
        _bsr_assign_padded_row_ranges,
        dim=dest.nrow,
        device=dest.device,
        inputs=[
            dest.nrow,
            src.offsets,
            src.row_ends,
            dest.offsets,
            dest.row_ends,
            dest.columns,
            status,
        ],
    )

    if check_status:
        _bsr_raise_if_status_error(status)

    wp.launch(
        _bsr_assign_padded_copy_blocks,
        dim=(src.nnz, *src.block_shape),
        device=dest.device,
        inputs=[
            src.scalar_type(src_scale),
            structure_only,
            dest.nrow,
            src.offsets,
            src.row_ends,
            src.columns,
            src.scalar_values,
            dest.offsets,
            dest.row_ends,
            dest.columns,
            dest.scalar_values,
        ],
    )


def _bsr_assign_padded_reblock(
    dest: BsrMatrix,
    src: BsrMatrix,
    src_scale: float,
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    structure_only: bool,
    status: wp.array | None = None,
    check_status: bool = True,
):
    if status is None:
        status = _bsr_temp_status(dest.device)
    else:
        status.zero_()

    wp.launch(
        _bsr_assign_padded_reblock_topology,
        dim=dest.nrow,
        device=dest.device,
        inputs=[
            src_subrows,
            src_subcols,
            dest_subrows,
            dest_subcols,
            src.nrow,
            dest.nrow,
            dest.ncol,
            src.offsets,
            src.row_ends,
            src.columns,
            dest.offsets,
            dest.row_ends,
            dest.columns,
            status,
        ],
    )

    if check_status:
        _bsr_raise_if_status_error(status)

    if not structure_only:
        dest.values.zero_()
        wp.launch(
            _bsr_assign_copy_blocks,
            dim=(src.nnz, src_subrows, src_subcols),
            device=dest.device,
            inputs=[
                src.scalar_type(src_scale),
                src_subrows,
                src_subcols,
                dest_subrows,
                dest_subcols,
                src.nrow,
                src.offsets,
                src.row_ends,
                src.columns,
                src.scalar_values,
                dest.offsets,
                dest.row_ends,
                dest.columns,
                dest.scalar_values,
            ],
        )


def bsr_assign(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    src: BsrMatrixOrExpression[BlockType[Any, Any, Any]],
    structure_only: bool = False,
    masked: bool = False,
    topology: str | None = None,
    overflow: str = "error",
    status: wp.array | None = None,
):
    """Copy the content of the ``src`` BSR matrix to ``dest``.

    Args:
      src: Matrix to be copied.
      dest: Destination matrix. May have a different block shape or scalar type
        than ``src``, in which case the required casting will be performed.
      structure_only: If ``True``, only the non-zero indices are copied, and uninitialized value storage is allocated
        to accommodate at least ``src.nnz`` blocks. If ``structure_only`` is ``False``, values are also copied with implicit
        casting if the two matrices use distinct scalar types.
      masked: If ``True``, keep the non-zero topology of ``dest`` unchanged.
      topology: Optional topology policy. ``"compact"`` keeps the existing
        compact rebuild behavior, ``"masked"`` is equivalent to ``masked=True``,
        and ``"padded"`` copies each source row into existing destination row
        capacity without changing ``dest.offsets``.
      overflow: Overflow policy for ``topology="padded"``. ``"error"``
        raises on insufficient row capacity, while ``"ignore"`` records status
        in ``status`` and leaves overflowing rows undefined.
      status: Optional single-element int array receiving asynchronous status
        for ``topology="padded"``. Required when ``overflow="ignore"``.
    """

    src, src_scale = _extract_matrix_and_scale(src)

    if dest.values.device != src.values.device:
        raise ValueError("Source and destination matrices must reside on the same device")

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if overflow not in ("error", "ignore"):
        raise NotImplementedError("Only overflow='error' and overflow='ignore' are currently implemented")
    if status is not None:
        _bsr_validate_status_array(status, dest.device)
        status.zero_()

    masked = topology == "masked"

    if src.block_shape[0] >= dest.block_shape[0]:
        src_subrows = src.block_shape[0] // dest.block_shape[0]
        dest_subrows = 1
    else:
        dest_subrows = dest.block_shape[0] // src.block_shape[0]
        src_subrows = 1

    if src_subrows * dest.block_shape[0] != src.block_shape[0] * dest_subrows:
        raise ValueError(
            f"Incompatible dest and src block shapes; block rows must evenly divide one another (Got {dest.block_shape[0]}, {src.block_shape[0]})"
        )

    if src.block_shape[1] >= dest.block_shape[1]:
        src_subcols = src.block_shape[1] // dest.block_shape[1]
        dest_subcols = 1
    else:
        dest_subcols = dest.block_shape[1] // src.block_shape[1]
        src_subcols = 1

    if src_subcols * dest.block_shape[1] != src.block_shape[1] * dest_subcols:
        raise ValueError(
            f"Incompatible dest and src block shapes; block columns must evenly divide one another (Got {dest.block_shape[1]}, {src.block_shape[1]})"
        )

    dest_nrow = (src.nrow * src_subrows) // dest_subrows
    dest_ncol = (src.ncol * src_subcols) // dest_subcols

    if src.nrow * src_subrows != dest_nrow * dest_subrows or src.ncol * src_subcols != dest_ncol * dest_subcols:
        raise ValueError(
            f"The requested block shape {dest.block_shape} does not evenly divide the source matrix of total size {src.shape}"
        )

    if topology == "padded":
        if overflow == "ignore" and status is None:
            raise ValueError("`status` must be supplied when using overflow='ignore'")

        if dest_nrow != dest.nrow or dest_ncol != dest.ncol:
            raise ValueError(
                f"Incompatible destination matrix size, expected ({dest_nrow}, {dest_ncol}), got ({dest.nrow}, {dest.ncol})"
            )

        if dest == src:
            if not structure_only and src_scale != 1.0:
                bsr_scale(dest, src_scale)
            return

        if dest.block_shape != src.block_shape:
            _bsr_assign_padded_reblock(
                dest=dest,
                src=src,
                src_scale=src_scale,
                src_subrows=src_subrows,
                src_subcols=src_subcols,
                dest_subrows=dest_subrows,
                dest_subcols=dest_subcols,
                structure_only=structure_only,
                status=status,
                check_status=overflow == "error",
            )
            return

        _bsr_assign_padded_same_block(
            dest=dest,
            src=src,
            src_scale=src_scale,
            structure_only=structure_only,
            status=status,
            check_status=overflow == "error",
        )
        return

    nnz_alloc = src.nnz * src_subrows * src_subcols
    if masked:
        if dest_nrow != dest.nrow or dest_ncol != dest.ncol:
            raise ValueError(
                f"Incompatible destination matrix size, expected ({dest_nrow}, {dest_ncol}), got ({dest.nrow}, {dest.ncol})"
            )
    else:
        _bsr_resize(dest, rows_of_blocks=dest_nrow, cols_of_blocks=dest_ncol)

    if dest.block_shape == src.block_shape and not masked:
        # Direct copy

        wp.copy(dest=dest.offsets, src=src.offsets, count=src.nrow + 1)
        wp.copy(dest=dest.row_ends, src=src.row_ends, count=src.nrow)
        dest.notify_nnz_changed(nnz=nnz_alloc)

        if nnz_alloc > 0:
            wp.copy(dest=dest.columns, src=src.columns, count=nnz_alloc)

            if not structure_only:
                warp._src.utils.array_cast(out_array=dest.values, in_array=src.values, count=nnz_alloc)
                bsr_scale(dest, src_scale)

    else:
        if not masked:
            # Compute destination rows and columns
            dest_rows = wp.empty(nnz_alloc, dtype=int, device=dest.device)
            dest_cols = wp.empty(nnz_alloc, dtype=int, device=dest.device)
            wp.launch(
                _bsr_assign_list_blocks,
                dim=(src.nnz, src_subrows, src_subcols),
                device=dest.device,
                inputs=[
                    src_subrows,
                    src_subcols,
                    dest_subrows,
                    dest_subcols,
                    src.nrow,
                    src.offsets,
                    src.row_ends,
                    src.columns,
                    dest_rows,
                    dest_cols,
                ],
            )

            _bsr_ensure_fits(dest, nnz=nnz_alloc)

            # Compute destination offsets from triplets
            from warp._src.context import runtime  # noqa: PLC0415

            if dest.device.is_cpu:
                native_func = runtime.core.wp_bsr_matrix_from_triplets_host
            else:
                native_func = runtime.core.wp_bsr_matrix_from_triplets_device

            nnz_buf, nnz_event = dest._setup_nnz_transfer()
            with wp.ScopedDevice(dest.device):
                native_func(
                    dest.block_size,
                    0,  # scalar_size_in_bytes
                    dest.nrow,
                    dest.ncol,
                    nnz_alloc,
                    None,  # device nnz
                    ctypes.cast(dest_rows.ptr, ctypes.POINTER(ctypes.c_int32)),
                    ctypes.cast(dest_cols.ptr, ctypes.POINTER(ctypes.c_int32)),
                    None,  # triplet values
                    0,  # zero_value_mask
                    masked,
                    None,  # summed block offsets
                    None,  # summed block indices
                    ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                    ctypes.cast(dest.row_ends.ptr, ctypes.POINTER(ctypes.c_int32)),
                    ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                    _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
                    _optional_ctypes_event(nnz_event),
                )
            _bsr_set_compact_row_ends(dest)

        # copy block values
        if not structure_only:
            dest.values.zero_()
            wp.launch(
                _bsr_assign_copy_blocks,
                dim=(src.nnz, src_subrows, src_subcols),
                device=dest.device,
                inputs=[
                    src.scalar_type(src_scale),
                    src_subrows,
                    src_subcols,
                    dest_subrows,
                    dest_subcols,
                    src.nrow,
                    src.offsets,
                    src.row_ends,
                    src.columns,
                    src.scalar_values,
                    dest.offsets,
                    dest.row_ends,
                    dest.columns,
                    dest.scalar_values,
                ],
            )


def bsr_copy(
    A: BsrMatrixOrExpression,
    scalar_type: Scalar | None = None,
    block_shape: tuple[int, int] | None = None,
    structure_only: bool = False,
    topology: str = "compact",
):
    """Return a copy of matrix ``A``, possibly changing its scalar type.

    Args:
       A: Matrix to be copied.
       scalar_type: If provided, the returned matrix will use this scalar type instead of the one from ``A``.
       block_shape: If provided, the returned matrix will use blocks of this shape instead of the one from ``A``.
         Both dimensions of ``block_shape`` must be either a multiple or an exact divider of the ones from ``A``.
       structure_only: If ``True``, only the non-zeros indices are copied, and uninitialized value storage is allocated
         to accommodate at least ``src.nnz`` blocks. If ``structure_only`` is ``False``, values are also copied with implicit
         casting if the two matrices use distinct scalar types.
       topology: Topology policy for the copy. ``"compact"`` uses the existing
         compact copy behavior, while ``"padded"`` preserves row capacity.
    """
    src, src_scale = _extract_matrix_and_scale(A)

    if scalar_type is None:
        scalar_type = src.scalar_type
    if block_shape is None:
        block_shape = src.block_shape

    if block_shape == (1, 1):
        block_type = scalar_type
    else:
        block_type = wp.types.matrix(shape=block_shape, dtype=scalar_type)

    if topology == "padded":
        if src.block_shape[0] >= block_shape[0]:
            src_subrows = src.block_shape[0] // block_shape[0]
            dest_subrows = 1
        else:
            dest_subrows = block_shape[0] // src.block_shape[0]
            src_subrows = 1

        if src_subrows * block_shape[0] != src.block_shape[0] * dest_subrows:
            raise ValueError(
                f"Incompatible dest and src block shapes; block rows must evenly divide one another (Got {block_shape[0]}, {src.block_shape[0]})"
            )

        if src.block_shape[1] >= block_shape[1]:
            src_subcols = src.block_shape[1] // block_shape[1]
            dest_subcols = 1
        else:
            dest_subcols = block_shape[1] // src.block_shape[1]
            src_subcols = 1

        if src_subcols * block_shape[1] != src.block_shape[1] * dest_subcols:
            raise ValueError(
                f"Incompatible dest and src block shapes; block columns must evenly divide one another (Got {block_shape[1]}, {src.block_shape[1]})"
            )

        copy_nrow = (src.nrow * src_subrows) // dest_subrows
        copy_ncol = (src.ncol * src_subcols) // dest_subcols

        if src.nrow * src_subrows != copy_nrow * dest_subrows or src.ncol * src_subcols != copy_ncol * dest_subcols:
            raise ValueError(
                f"The requested block shape {block_shape} does not evenly divide the source matrix of total size {src.shape}"
            )

        copy = bsr_zeros(
            rows_of_blocks=copy_nrow,
            cols_of_blocks=copy_ncol,
            block_type=block_type,
            device=src.device,
        )
        copy.values.requires_grad = src.requires_grad

        if block_shape == src.block_shape:
            _bsr_ensure_fits(copy, nnz=src.nnz)
            wp.copy(dest=copy.offsets, src=src.offsets, count=src.nrow + 1)
            _bsr_assign_padded_same_block(
                dest=copy,
                src=src,
                src_scale=src_scale,
                structure_only=structure_only,
                check_status=False,
            )
        else:
            max_nnz = src.nnz * src_subrows * src_subcols
            _bsr_ensure_fits(copy, nnz=max_nnz)
            if copy.nrow > 0:
                wp.launch(
                    _bsr_copy_reblocked_capacity_counts,
                    dim=copy.nrow,
                    device=copy.device,
                    inputs=[
                        src_subrows,
                        src_subcols,
                        dest_subrows,
                        src.nrow,
                        copy.nrow,
                        src.offsets,
                        copy.offsets,
                    ],
                )
                warp._src.utils.array_scan(copy.offsets, copy.offsets, inclusive=True)
            _bsr_assign_padded_reblock(
                dest=copy,
                src=src,
                src_scale=src_scale,
                src_subrows=src_subrows,
                src_subcols=src_subcols,
                dest_subrows=dest_subrows,
                dest_subcols=dest_subcols,
                structure_only=structure_only,
                check_status=False,
            )
    else:
        copy = bsr_zeros(
            rows_of_blocks=src.nrow,
            cols_of_blocks=src.ncol,
            block_type=block_type,
            device=src.device,
        )
        copy.values.requires_grad = src.requires_grad
        bsr_assign(dest=copy, src=A, structure_only=structure_only, topology=topology)

    return copy


@wp.kernel
def _bsr_transpose_values(
    col_count: int,
    scale: Any,
    bsr_offsets: wp.array(dtype=int),
    bsr_row_ends: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_values: wp.array3d(dtype=Any),
    block_index_map: wp.array(dtype=int),
    transposed_bsr_offsets: wp.array(dtype=int),
    transposed_bsr_row_ends: wp.array(dtype=int),
    transposed_bsr_columns: wp.array(dtype=int),
    transposed_bsr_values: wp.array3d(dtype=Any),
):
    block, i, j = wp.tid()

    if block >= transposed_bsr_offsets[col_count]:
        return

    if block_index_map:
        src_block = block_index_map[block]
    else:
        row = _bsr_row_index_active(transposed_bsr_offsets, col_count, block, transposed_bsr_row_ends)
        col = transposed_bsr_columns[block]
        src_block = _bsr_block_index_active(col, row, bsr_offsets, bsr_columns, bsr_row_ends)
        if src_block == -1:
            return

    transposed_bsr_values[block, i, j] = bsr_values[src_block, j, i] * scale


@wp.kernel(enable_backward=False)
def _bsr_set_transpose_padded_count(
    src_row_count: int,
    dest_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    dest_row = wp.tid()

    if dest_row >= dest_row_count:
        return

    block_count = int(0)
    for src_row in range(src_row_count):
        if _bsr_block_index_active(src_row, dest_row, src_offsets, src_columns, src_row_ends) != -1:
            block_count += 1

    dest_beg = dest_offsets[dest_row]
    dest_capacity_end = dest_offsets[dest_row + 1]
    if dest_beg + block_count > dest_capacity_end:
        dest_row_ends[dest_row] = dest_beg
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
        return

    dest_active_end = dest_beg + block_count
    dest_row_ends[dest_row] = dest_active_end

    for block in range(dest_active_end, dest_capacity_end):
        dest_columns[block] = -1


@wp.kernel
def _bsr_set_transpose_padded_values(
    scale: Any,
    src_row_count: int,
    dest_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_row_ends: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    dest_row, i, j = wp.tid()

    if dest_row >= dest_row_count:
        return

    write = int(dest_offsets[dest_row])
    dest_end = dest_row_ends[dest_row]

    for src_row in range(src_row_count):
        if write >= dest_end:
            return

        src_block = _bsr_block_index_active(src_row, dest_row, src_offsets, src_columns, src_row_ends)
        if src_block != -1:
            if i == 0 and j == 0:
                dest_columns[write] = src_row
            dest_values[write, i, j] = src_values[src_block, j, i] * scale
            write += 1


def bsr_set_transpose(
    dest: BsrMatrix[BlockType[Cols, Rows, Scalar]],
    src: BsrMatrixOrExpression[BlockType[Rows, Cols, Scalar]],
    masked: bool = False,
    topology: str | None = None,
    overflow: str = "error",
    status: wp.array | None = None,
):
    """Assign the transposed matrix ``src`` to matrix ``dest``.

    Args:
        dest: Sparse matrix to populate.
        src: Sparse matrix to transpose.
        masked: If ``True``, keep the non-zero topology of ``dest`` unchanged.
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          ``masked=True``, and ``"padded"`` writes the transposed active
          topology into existing destination row capacity.
        overflow: Overflow policy for ``topology="padded"``. ``"error"``
          raises on insufficient row capacity, while ``"ignore"`` records
          status in ``status`` and leaves overflowing rows undefined.
        status: Optional single-element int array receiving asynchronous status
          for ``topology="padded"``. Required when ``overflow="ignore"``.
    """

    src, src_scale = _extract_matrix_and_scale(src)

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if overflow not in ("error", "ignore"):
        raise NotImplementedError("Only overflow='error' and overflow='ignore' are currently implemented")
    if status is not None:
        _bsr_validate_status_array(status, dest.device)
        status.zero_()

    masked = topology == "masked"

    if dest.values.device != src.values.device:
        raise ValueError(
            f"All arguments must reside on the same device, got {dest.values.device} and {src.values.device}"
        )

    if dest.scalar_type != src.scalar_type:
        raise ValueError(f"All arguments must have the same scalar type, got {dest.scalar_type} and {src.scalar_type}")

    transpose_block_shape = src.block_shape[::-1]

    if dest.block_shape != transpose_block_shape:
        raise ValueError(f"Destination block shape must be {transpose_block_shape}, got {dest.block_shape}")

    if topology == "padded":
        if overflow == "ignore" and status is None:
            raise ValueError("`status` must be supplied when using overflow='ignore'")

        if dest.nrow != src.ncol or dest.ncol != src.nrow:
            raise ValueError(
                f"Destination matrix must have {src.ncol} rows and {src.nrow} columns, got {dest.nrow} and {dest.ncol}"
            )

        check_status = overflow == "error"
        if status is None:
            status = _bsr_temp_status(dest.device)

        wp.launch(
            _bsr_set_transpose_padded_count,
            dim=dest.nrow,
            device=dest.device,
            inputs=[
                src.nrow,
                dest.nrow,
                src.offsets,
                src.row_ends,
                src.columns,
                dest.offsets,
                dest.row_ends,
                dest.columns,
                status,
            ],
        )

        if check_status:
            _bsr_raise_if_status_error(status)

        wp.launch(
            _bsr_set_transpose_padded_values,
            dim=(dest.nrow, *dest.block_shape),
            device=dest.device,
            inputs=[
                dest.scalar_type(src_scale),
                src.nrow,
                dest.nrow,
                src.offsets,
                src.row_ends,
                src.columns,
                src.scalar_values,
                dest.offsets,
                dest.row_ends,
                dest.columns,
                dest.scalar_values,
            ],
        )
        return

    if masked:
        if dest.nrow != src.ncol or dest.ncol != src.nrow:
            raise ValueError(
                f"Destination matrix must have {src.ncol} rows and {src.nrow} columns, got {dest.nrow} and {dest.ncol}"
            )
        block_index_map = None
        dest.values.zero_()
    else:
        _bsr_resize(dest, rows_of_blocks=src.ncol, cols_of_blocks=src.nrow)

        nnz = src.nnz
        if nnz == 0:
            bsr_set_zero(dest)
            return

        # Increase dest array sizes if needed
        _bsr_ensure_fits(dest, nnz=nnz)

        from warp._src.context import runtime  # noqa: PLC0415

        if dest.values.device.is_cpu:
            native_func = runtime.core.wp_bsr_transpose_host
        else:
            native_func = runtime.core.wp_bsr_transpose_device

        block_index_map = wp.empty(shape=2 * nnz, dtype=int, device=src.device)

        with wp.ScopedDevice(dest.device):
            native_func(
                src.nrow,
                src.ncol,
                nnz,
                ctypes.cast(src.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(src.row_ends.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(src.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(block_index_map.ptr, ctypes.POINTER(ctypes.c_int32)),
            )

            dest._copy_nnz_async()
            _bsr_set_compact_row_ends(dest)

    wp.launch(
        _bsr_transpose_values,
        dim=(dest.nnz, *dest.block_shape),
        device=dest.device,
        inputs=[
            src.ncol,
            dest.scalar_type(src_scale),
            src.offsets,
            src.row_ends,
            src.columns,
            src.scalar_values,
            block_index_map,
            dest.offsets,
            dest.row_ends,
            dest.columns,
        ],
        outputs=[dest.scalar_values],
    )


def bsr_transposed(A: BsrMatrixOrExpression) -> BsrMatrix:
    """Return a copy of the transposed matrix ``A``."""

    if A.block_shape == (1, 1):
        block_type = A.values.dtype
    else:
        block_type = wp.types.matrix(shape=A.block_shape[::-1], dtype=A.scalar_type)

    transposed = bsr_zeros(
        rows_of_blocks=A.ncol,
        cols_of_blocks=A.nrow,
        block_type=block_type,
        device=A.device,
    )
    transposed.values.requires_grad = A.requires_grad
    bsr_set_transpose(dest=transposed, src=A)
    return transposed


@wp.kernel
def _bsr_get_diag_kernel(
    scale: Any,
    A_offsets: wp.array(dtype=int),
    A_row_ends: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array3d(dtype=Any),
    out: wp.array3d(dtype=Any),
):
    row, br, bc = wp.tid()

    diag = _bsr_block_index_active(row, row, A_offsets, A_columns, A_row_ends)
    if diag != -1:
        out[row, br, bc] = scale * A_values[diag, br, bc]


def bsr_get_diag(A: BsrMatrixOrExpression[BlockType], out: Array[BlockType] | None = None) -> Array[BlockType]:
    """Return the array of blocks that constitute the diagonal of a sparse matrix.

    Args:
        A: The sparse matrix from which to extract the diagonal.
        out: If provided, the array into which to store the diagonal blocks.
    """

    A, scale = _extract_matrix_and_scale(A)

    dim = min(A.nrow, A.ncol)

    if out is None:
        out = wp.zeros(shape=(dim,), dtype=A.values.dtype, device=A.values.device)
    else:
        if not types_equal(out.dtype, A.values.dtype):
            raise ValueError(f"Output array must have type {A.values.dtype}, got {out.dtype}")
        if out.device != A.values.device:
            raise ValueError(f"Output array must reside on device {A.values.device}, got {out.device}")
        if out.shape[0] < dim:
            raise ValueError(f"Output array must be of length at least {dim}, got {out.shape[0]}")
        out.zero_()

    wp.launch(
        kernel=_bsr_get_diag_kernel,
        dim=(dim, *A.block_shape),
        device=A.values.device,
        inputs=[
            A.scalar_type(scale),
            A.offsets,
            A.row_ends,
            A.columns,
            A.scalar_values,
            _as_3d_array(out, A.block_shape),
        ],
    )

    return out


@wp.kernel(enable_backward=False)
def _bsr_set_diag_kernel(
    nnz: int,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
):
    row = wp.tid()
    A_offsets[row] = wp.min(row, nnz)
    if row < nnz:
        A_columns[row] = row


def bsr_set_diag(
    A: BsrMatrix[BlockType],
    diag: BlockType | Array[BlockType],
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
) -> None:
    """Set ``A`` as a block-diagonal matrix.

    Args:
        A: The sparse matrix to modify.
        diag: Specifies the values for diagonal blocks. Can be one of:

          - A Warp array of type ``A.values.dtype``: Each element defines one block of the diagonal
          - A constant value of type ``A.values.dtype``: This value is assigned to all diagonal blocks
          - ``None``: Diagonal block values are left uninitialized

        rows_of_blocks: If not ``None``, the new number of rows of blocks.
        cols_of_blocks: If not ``None``, the new number of columns of blocks.

    The shape of the matrix will be defined one of the following, in this order:

    - ``rows_of_blocks`` and ``cols_of_blocks``, if provided.
      If only one is given, the second is assumed equal.
    - The first dimension of ``diag``, if ``diag`` is an array
    - The current dimensions of ``A`` otherwise
    """

    if rows_of_blocks is None and cols_of_blocks is not None:
        rows_of_blocks = cols_of_blocks
    if cols_of_blocks is None and rows_of_blocks is not None:
        cols_of_blocks = rows_of_blocks

    if is_array(diag):
        if rows_of_blocks is None:
            rows_of_blocks = diag.shape[0]
            cols_of_blocks = diag.shape[0]

    if rows_of_blocks is not None:
        _bsr_resize(A, rows_of_blocks, cols_of_blocks)

    nnz = min(A.nrow, A.ncol)
    A.notify_nnz_changed(nnz=nnz)  # notify change of nnz upper bound

    wp.launch(
        kernel=_bsr_set_diag_kernel,
        dim=A.nrow + 1,
        device=A.offsets.device,
        inputs=[nnz, A.offsets, A.columns],
    )
    _bsr_set_compact_row_ends(A)

    A.notify_nnz_changed(nnz=nnz)  # notify change of offsets

    if is_array(diag):
        wp.copy(src=diag, dest=A.values, count=nnz)
    elif diag is not None:
        A.values.fill_(diag)


def bsr_diag(
    diag: BlockType | Array[BlockType] | None = None,
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
    block_type: BlockType | None = None,
    device=None,
) -> BsrMatrix[BlockType]:
    """Create and return a block-diagonal BSR matrix from an given block value or array of block values.

    Args:
        diag: Specifies the values for diagonal blocks. Can be one of:

          - A Warp array of type ``A.values.dtype``: Each element defines one block of the diagonal
          - A constant value of type ``A.values.dtype``: This value is assigned to all diagonal blocks
        rows_of_blocks: If not ``None``, the new number of rows of blocks
        cols_of_blocks: If not ``None``, the new number of columns of blocks
        block_type: If ``diag`` is ``None``, block type of the matrix. Otherwise deduced from ``diag``
        device: If ``diag`` is not a Warp array, device on which to allocate the matrix. Otherwise deduced from ``diag``

    The shape of the matrix will be defined one of the following, in this order:

    - ``rows_of_blocks`` and ``cols_of_blocks``, if provided.
      If only one is given, the second is assumed equal.
    - The first dimension of ``diag`` if ``diag`` is an array.
    """

    if rows_of_blocks is None and cols_of_blocks is not None:
        rows_of_blocks = cols_of_blocks
    if cols_of_blocks is None and rows_of_blocks is not None:
        cols_of_blocks = rows_of_blocks

    if is_array(diag):
        if rows_of_blocks is None:
            rows_of_blocks = diag.shape[0]
            cols_of_blocks = diag.shape[0]

        block_type = diag.dtype
        device = diag.device
    else:
        if rows_of_blocks is None:
            raise ValueError(
                "rows_of_blocks and/or cols_of_blocks must be provided for constructing a diagonal matrix with uniform diagonal"
            )

    if block_type is None:
        if diag is None:
            raise ValueError("Either `diag` or `block_type` needs to be provided")

        block_type = type(diag)
        if not type_is_matrix(block_type) and len(getattr(diag, "shape", ())) == 2:
            block_type = wp.types.matrix(shape=diag.shape, dtype=diag.dtype)

    A = bsr_zeros(rows_of_blocks, cols_of_blocks, block_type=block_type, device=device)
    if is_array(diag):
        A.values.requires_grad = diag.requires_grad
    bsr_set_diag(A, diag)
    return A


def bsr_set_identity(A: BsrMatrix, rows_of_blocks: int | None = None) -> None:
    """Set ``A`` as the identity matrix.

    Args:
        A: The sparse matrix to modify.
        rows_of_blocks: If provided, the matrix will be resized as a square
          matrix with ``rows_of_blocks`` rows and columns.
    """

    if A.block_shape == (1, 1):
        identity = A.scalar_type(1.0)
    else:
        identity = eye(A.block_shape[0])

    bsr_set_diag(A, diag=identity, rows_of_blocks=rows_of_blocks, cols_of_blocks=rows_of_blocks)


def bsr_identity(
    rows_of_blocks: int,
    block_type: BlockType[Rows, Rows, Scalar],
    device: wp.DeviceLike = None,
) -> BsrMatrix[BlockType[Rows, Rows, Scalar]]:
    """Create and return a square identity matrix.

    Args:
        rows_of_blocks: Number of rows and columns of blocks in the created matrix.
        block_type: Block type for the newly created matrix. Must be square
        device: Device onto which to allocate the data arrays
    """
    A = bsr_zeros(
        rows_of_blocks=rows_of_blocks,
        cols_of_blocks=rows_of_blocks,
        block_type=block_type,
        device=device,
    )
    bsr_set_identity(A)
    return A


@wp.kernel
def _bsr_scale_kernel(
    alpha: Any,
    values: wp.array(dtype=Any),
):
    row = wp.tid()
    values[row] = alpha * values[row]


@wp.kernel
def _bsr_scale_kernel(
    alpha: Any,
    values: wp.array3d(dtype=Any),
):
    row, br, bc = wp.tid()
    values[row, br, bc] = alpha * values[row, br, bc]


def bsr_scale(x: BsrMatrixOrExpression, alpha: Scalar) -> BsrMatrix:
    """Perform the operation ``x := alpha * x`` on BSR matrix ``x`` and return ``x``."""

    x, scale = _extract_matrix_and_scale(x)
    alpha *= scale

    if alpha != 1.0 and x.nnz > 0:
        if alpha == 0.0:
            x.values.zero_()
        else:
            alpha = x.scalar_type(alpha)

            wp.launch(
                kernel=_bsr_scale_kernel,
                dim=(x.nnz, *x.block_shape),
                device=x.values.device,
                inputs=[alpha, x.scalar_values],
            )

    return x


@wp.kernel(enable_backward=False)
def _bsr_get_block_row(
    row_count: int, bsr_offsets: wp.array(dtype=int), bsr_row_ends: wp.array(dtype=int), rows: wp.array(dtype=int)
):
    block = wp.tid()
    rows[block] = _bsr_row_index_active(bsr_offsets, row_count, block, bsr_row_ends)


@wp.kernel
def _bsr_axpy_add_block(
    src_offset: int,
    scale: Any,
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    dst_offsets: wp.array(dtype=int),
    dst_row_ends: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dst_values: wp.array3d(dtype=Any),
):
    i, br, bc = wp.tid()
    row = rows[i + src_offset]
    col = cols[i + src_offset]

    block = _bsr_block_index_active(row, col, dst_offsets, dst_columns, dst_row_ends)
    if block != -1:
        dst_values[block, br, bc] += scale * src_values[i, br, bc]


@wp.kernel
def _bsr_axpy_masked(
    alpha: Any,
    row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_ends: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dst_offsets: wp.array(dtype=int),
    dst_row_ends: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
    dst_values: wp.array3d(dtype=Any),
):
    block, br, bc = wp.tid()

    row = _bsr_row_index_active(dst_offsets, row_count, block, dst_row_ends)
    if row == -1:
        return

    col = dst_columns[block]
    src_block = _bsr_block_index_active(row, col, src_offsets, src_columns, src_row_ends)
    if src_block != -1:
            dst_values[block, br, bc] += alpha * src_values[src_block, br, bc]


@wp.kernel(enable_backward=False)
def _bsr_axpy_padded_count(
    row_count: int,
    x_offsets: wp.array(dtype=int),
    x_row_ends: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_row_ends: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    row_block_counts: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    if row >= row_count:
        return

    x_block = int(x_offsets[row])
    x_end = x_row_ends[row]
    y_block = int(y_offsets[row])
    y_end = y_row_ends[row]

    block_count = int(0)

    while x_block < x_end and y_block < y_end:
        x_col = x_columns[x_block]
        y_col = y_columns[y_block]

        block_count += 1
        if x_col == y_col:
            x_block += 1
            y_block += 1
        elif x_col < y_col:
            x_block += 1
        else:
            y_block += 1

    block_count += x_end - x_block
    block_count += y_end - y_block

    if y_offsets[row] + block_count > y_offsets[row + 1]:
        row_block_counts[row] = -1
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
    else:
        row_block_counts[row] = block_count


@wp.kernel
def _bsr_axpy_padded_fill(
    alpha: Any,
    beta: Any,
    row_count: int,
    x_offsets: wp.array(dtype=int),
    x_row_ends: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    x_values: wp.array3d(dtype=Any),
    y_offsets: wp.array(dtype=int),
    y_row_ends: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    y_values: wp.array3d(dtype=Any),
    row_block_counts: wp.array(dtype=int),
):
    row, br, bc = wp.tid()

    if row >= row_count:
        return

    block_count = row_block_counts[row]
    if block_count < 0:
        return

    x_block = x_row_ends[row] - 1
    x_beg = x_offsets[row]
    y_block = y_row_ends[row] - 1
    y_beg = y_offsets[row]
    write = int(y_beg + block_count - 1)

    while write >= y_beg:
        use_x = bool(False)
        use_y = bool(False)
        col = int(0)

        if x_block >= x_beg and y_block >= y_beg:
            x_col = x_columns[x_block]
            y_col = y_columns[y_block]
            if x_col == y_col:
                use_x = True
                use_y = True
                col = x_col
            elif x_col > y_col:
                use_x = True
                col = x_col
            else:
                use_y = True
                col = y_col
        elif x_block >= x_beg:
            use_x = True
            col = x_columns[x_block]
        else:
            use_y = True
            col = y_columns[y_block]

        value = y_values.dtype(0.0)
        if use_x:
            value += alpha * x_values[x_block, br, bc]
            x_block -= 1
        if use_y:
            value += beta * y_values[y_block, br, bc]
            y_block -= 1

        if br == 0 and bc == 0:
            y_columns[write] = col
        y_values[write, br, bc] = value

        write -= 1


@wp.kernel(enable_backward=False)
def _bsr_axpy_padded_finalize(
    row_count: int,
    y_offsets: wp.array(dtype=int),
    y_row_ends: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    row_block_counts: wp.array(dtype=int),
):
    row = wp.tid()

    if row >= row_count:
        return

    row_beg = y_offsets[row]
    capacity_end = y_offsets[row + 1]
    block_count = row_block_counts[row]

    if block_count < 0:
        y_row_ends[row] = row_beg
        block_count = int(0)
    else:
        y_row_ends[row] = row_beg + block_count

    for block in range(row_beg + block_count, capacity_end):
        y_columns[block] = -1


class bsr_axpy_work_arrays(_BsrStatusMixin):
    """Opaque structure for persisting :func:`bsr_axpy` temporary work buffers across calls."""

    def __init__(self):
        self._reset(None)

    def _reset(self, device):
        self.device = device
        self._reset_status()
        self._sum_rows = None
        self._sum_cols = None
        self._old_y_values = None
        self._old_x_values = None

    def _allocate(self, device, y: BsrMatrix, sum_nnz: int):
        if self.device != device:
            self._reset(device)

        if self._sum_rows is None or self._sum_rows.size < sum_nnz:
            self._sum_rows = wp.empty(shape=(sum_nnz), dtype=int, device=self.device)
        if self._sum_cols is None or self._sum_cols.size < sum_nnz:
            self._sum_cols = wp.empty(shape=(sum_nnz), dtype=int, device=self.device)

        if self._old_y_values is None or self._old_y_values.size < y.nnz:
            self._old_y_values = wp.empty_like(y.values[: y.nnz])


def bsr_axpy(
    x: BsrMatrixOrExpression,
    y: BsrMatrix[BlockType[Rows, Cols, Scalar]] | None = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 1.0,
    masked: bool = False,
    work_arrays: bsr_axpy_work_arrays | None = None,
    topology: str | None = None,
    overflow: str = "error",
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """Perform the sparse matrix addition ``y := alpha * X + beta * y`` on BSR matrices ``x`` and ``y`` and return ``y``.

    The ``x`` and ``y`` matrices are allowed to alias.

    Args:
        x: Read-only first operand.
        y: Mutable second operand and output matrix. If ``y`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for ``x``.
        beta: Uniform scaling factor for ``y``.
        masked: If ``True``, keep the non-zero topology of ``y`` unchanged.
        work_arrays: In most cases, this function will require the use of temporary storage.
          This storage can be reused across calls by passing an instance of
          :class:`bsr_axpy_work_arrays` in ``work_arrays``.
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          ``masked=True``, and ``"padded"`` writes the result topology into
          existing destination row capacity.
        overflow: Overflow policy for ``topology="padded"``. ``"error"``
          raises on insufficient row capacity, while ``"ignore"`` records
          status in supplied ``work_arrays`` and leaves overflowing rows
          undefined.
    """

    x, x_scale = _extract_matrix_and_scale(x)
    alpha *= x_scale

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if overflow not in ("error", "ignore"):
        raise NotImplementedError("Only overflow='error' and overflow='ignore' are currently implemented")

    masked = topology == "masked"

    if y is None:
        if masked or topology == "padded":
            raise ValueError("Left-hand-side 'y' matrix must be provided for this topology policy")

        # If not output matrix is provided, allocate it for convenience
        y = bsr_zeros(x.nrow, x.ncol, block_type=x.values.dtype, device=x.values.device)
        y.values.requires_grad = x.requires_grad
        beta = 0.0

    x_nnz = x.nnz
    y_nnz = y.nnz

    if topology == "padded":
        if overflow == "ignore" and work_arrays is None:
            raise ValueError("`work_arrays` must be supplied when using overflow='ignore'")

        if x.values.device != y.values.device:
            raise ValueError(
                f"All arguments must reside on the same device, got {x.values.device} and {y.values.device}"
            )

        if x.scalar_type != y.scalar_type or x.block_shape != y.block_shape:
            raise ValueError(
                f"Matrices must have the same block type, got ({x.block_shape}, {x.scalar_type}) and ({y.block_shape}, {y.scalar_type})"
            )

        if x.nrow != y.nrow or x.ncol != y.ncol:
            raise ValueError(
                f"Matrices must have the same number of rows and columns, got ({x.nrow}, {x.ncol}) and ({y.nrow}, {y.ncol})"
            )

        if beta == 0.0:
            status = work_arrays._ensure_status(y.device) if work_arrays is not None else None
            bsr_assign(dest=y, src=x, topology="padded", overflow=overflow, status=status)
            return bsr_scale(y, alpha=alpha)

        if alpha == 0.0 or x_nnz == 0:
            return bsr_scale(y, alpha=beta)

        if x == y:
            return bsr_scale(y, alpha=alpha + beta)

        if not isinstance(alpha, y.scalar_type):
            alpha = y.scalar_type(alpha)
        if not isinstance(beta, y.scalar_type):
            beta = y.scalar_type(beta)

        if work_arrays is None:
            work_arrays = bsr_axpy_work_arrays()

        work_arrays._allocate(y.device, y, max(x_nnz + y_nnz, y.nrow))
        status = work_arrays._ensure_status(y.device)
        row_block_counts = work_arrays._sum_rows

        wp.launch(
            _bsr_axpy_padded_count,
            dim=y.nrow,
            device=y.device,
            inputs=[
                y.nrow,
                x.offsets,
                x.row_ends,
                x.columns,
                y.offsets,
                y.row_ends,
                y.columns,
                row_block_counts,
                status,
            ],
        )

        if overflow == "error":
            _bsr_raise_if_status_error(status)

        wp.launch(
            _bsr_axpy_padded_fill,
            dim=(y.nrow, *y.block_shape),
            device=y.device,
            inputs=[
                alpha,
                beta,
                y.nrow,
                x.offsets,
                x.row_ends,
                x.columns,
                x.scalar_values,
                y.offsets,
                y.row_ends,
                y.columns,
                y.scalar_values,
                row_block_counts,
            ],
        )

        wp.launch(
            _bsr_axpy_padded_finalize,
            dim=y.nrow,
            device=y.device,
            inputs=[
                y.nrow,
                y.offsets,
                y.row_ends,
                y.columns,
                row_block_counts,
            ],
        )
        return y

    # Handle easy cases first
    if beta == 0.0 or y_nnz == 0:
        bsr_assign(src=x, dest=y, masked=masked)
        return bsr_scale(y, alpha=alpha)

    if alpha == 0.0 or x_nnz == 0:
        return bsr_scale(y, alpha=beta)

    if x == y:
        # Aliasing case
        return bsr_scale(y, alpha=alpha + beta)

    # General case

    if not isinstance(alpha, y.scalar_type):
        alpha = y.scalar_type(alpha)
    if not isinstance(beta, y.scalar_type):
        beta = y.scalar_type(beta)

    if x.values.device != y.values.device:
        raise ValueError(f"All arguments must reside on the same device, got {x.values.device} and {y.values.device}")

    if x.scalar_type != y.scalar_type or x.block_shape != y.block_shape:
        raise ValueError(
            f"Matrices must have the same block type, got ({x.block_shape}, {x.scalar_type}) and ({y.block_shape}, {y.scalar_type})"
        )

    if x.nrow != y.nrow or x.ncol != y.ncol:
        raise ValueError(
            f"Matrices must have the same number of rows and columns, got ({x.nrow}, {x.ncol}) and ({y.nrow}, {y.ncol})"
        )

    device = y.values.device
    if masked:
        bsr_scale(y, alpha=beta.value)
        wp.launch(
            kernel=_bsr_axpy_masked,
            device=device,
            dim=(y_nnz, y.block_shape[0], y.block_shape[1]),
            inputs=[
                alpha,
                x.nrow,
                x.offsets,
                x.row_ends,
                x.columns,
                x.scalar_values,
                y.offsets,
                y.row_ends,
                y.columns,
                y.scalar_values,
            ],
        )

    else:
        if work_arrays is None:
            work_arrays = bsr_axpy_work_arrays()

        sum_nnz = x_nnz + y_nnz
        work_arrays._allocate(device, y, sum_nnz)

        wp.copy(work_arrays._sum_cols, y.columns, 0, 0, y_nnz)
        y.uncompress_rows(out=work_arrays._sum_rows)

        wp.copy(work_arrays._sum_cols, x.columns, y_nnz, 0, x_nnz)
        x.uncompress_rows(out=work_arrays._sum_rows[y_nnz:])

        # Save old y values before overwriting matrix
        wp.copy(dest=work_arrays._old_y_values, src=y.values, count=y.nnz)

        # Increase dest array sizes if needed
        _bsr_ensure_fits(y, nnz=sum_nnz)

        from warp._src.context import runtime  # noqa: PLC0415

        if device.is_cpu:
            native_func = runtime.core.wp_bsr_matrix_from_triplets_host
        else:
            native_func = runtime.core.wp_bsr_matrix_from_triplets_device

        old_y_nnz = y_nnz
        nnz_buf, nnz_event = y._setup_nnz_transfer()

        with wp.ScopedDevice(y.device):
            native_func(
                y.block_size,
                0,  # scalar_size_in_bytes
                y.nrow,
                y.ncol,
                sum_nnz,
                None,  # device nnz
                ctypes.cast(work_arrays._sum_rows.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(work_arrays._sum_cols.ptr, ctypes.POINTER(ctypes.c_int32)),
                None,  # triplet values
                0,  # zero_value_mask
                masked,
                None,  # summed block offsets
                None,  # summed block indices
                ctypes.cast(y.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(y.row_ends.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(y.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
                _optional_ctypes_event(nnz_event),
            )
        _bsr_set_compact_row_ends(y)

        y.values.zero_()

        wp.launch(
            kernel=_bsr_axpy_add_block,
            device=device,
            dim=(old_y_nnz, y.block_shape[0], y.block_shape[1]),
            inputs=[
                0,
                beta,
                work_arrays._sum_rows,
                work_arrays._sum_cols,
                y.offsets,
                y.row_ends,
                y.columns,
                _as_3d_array(work_arrays._old_y_values, y.block_shape),
                y.scalar_values,
            ],
        )

        wp.launch(
            kernel=_bsr_axpy_add_block,
            device=device,
            dim=(x_nnz, y.block_shape[0], y.block_shape[1]),
            inputs=[
                old_y_nnz,
                alpha,
                work_arrays._sum_rows,
                work_arrays._sum_cols,
                y.offsets,
                y.row_ends,
                y.columns,
                x.scalar_values,
                y.scalar_values,
            ],
        )

    return y


def make_bsr_mm_count_coeffs(tile_size):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=tile_size)
    def bsr_mm_count_coeffs(
        y_ncol: int,
        z_nnz: int,
        x_offsets: wp.array(dtype=int),
        x_row_ends: wp.array(dtype=int),
        x_columns: wp.array(dtype=int),
        y_offsets: wp.array(dtype=int),
        y_row_ends: wp.array(dtype=int),
        y_columns: wp.array(dtype=int),
        row_min: wp.array(dtype=int),
        block_counts: wp.array(dtype=int),
    ):
        row, lane = wp.tid()
        row_count = int(0)

        x_beg = x_offsets[row]
        x_end = x_row_ends[row]

        min_col = y_ncol
        max_col = int(0)

        for x_block in range(x_beg + lane, x_end, tile_size):
            x_col = x_columns[x_block]
            y_row_beg = y_offsets[x_col]
            y_row_end = y_row_ends[x_col]
            block_count = y_row_end - y_row_beg
            if block_count != 0:
                min_col = wp.min(y_columns[y_row_beg], min_col)
                max_col = wp.max(y_columns[y_row_end - 1], max_col)

            block_counts[x_block + 1] = block_count
            row_count += block_count

        if wp.static(tile_size) > 1:
            row_count = wp.tile_sum(wp.tile(row_count))[0]
            min_col = wp.tile_min(wp.tile(min_col))[0]
            max_col = wp.tile_max(wp.tile(max_col))[0]
        col_range_size = wp.max(0, max_col - min_col + 1)

        if row_count > col_range_size:
            # Optimization for deep products.
            # Do not store the whole whole list of src product terms, they would be highly redundant
            # Instead just mark a range in the output matrix

            if lane == 0:
                row_min[row] = min_col
                block_counts[x_end] = col_range_size

            for x_block in range(x_beg + lane, x_end - 1, tile_size):
                block_counts[x_block + 1] = 0
        elif lane == 0:
            row_min[row] = -1

        if lane == 0 and row == 0:
            block_counts[0] = z_nnz

    return bsr_mm_count_coeffs


@wp.kernel(enable_backward=False)
def _bsr_mm_list_coeffs(
    copied_z_nnz: int,
    mm_nnz: int,
    x_nrow: int,
    x_offsets: wp.array(dtype=int),
    x_row_ends: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_row_ends: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    mm_row_min: wp.array(dtype=int),
    mm_offsets: wp.array(dtype=int),
    mm_rows: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_src_blocks: wp.array(dtype=int),
):
    mm_block = wp.tid() + copied_z_nnz

    x_nnz = x_offsets[x_nrow]

    x_block = bsr_row_index(mm_offsets, x_nnz, mm_block)

    if x_block == -1:
        mm_cols[mm_block] = -1
        mm_rows[mm_block] = -1
        return

    if mm_block + 1 == mm_nnz and mm_nnz < mm_offsets[x_nnz]:
        wp.printf(
            "Number of potential `bsr_mm` blocks (%d) exceeded `max_nnz` (%d)\n",
            mm_offsets[x_nnz] - copied_z_nnz,
            mm_nnz - copied_z_nnz,
        )

    pos = mm_block - mm_offsets[x_block]

    row = _bsr_row_index_active(x_offsets, x_nrow, x_block, x_row_ends)

    row_min_col = mm_row_min[row]
    if row_min_col == -1:
        x_col = x_columns[x_block]
        y_beg = y_offsets[x_col]
        y_block = y_beg + pos
        col = y_columns[y_block]
        src_block = x_block
    else:
        col = row_min_col + pos
        src_block = -1

    mm_cols[mm_block] = col
    mm_rows[mm_block] = row
    mm_src_blocks[mm_block] = src_block


@wp.func
def _bsr_mm_use_triplets(
    row: int,
    mm_block: int,
    mm_row_min: wp.array(dtype=int),
    row_offsets: wp.array(dtype=int),
    row_ends: wp.array(dtype=int),
    summed_triplet_offsets: wp.array(dtype=int),
):
    x_beg = row_offsets[row]
    x_end = row_ends[row]

    if mm_row_min:
        if mm_row_min[row] == -1:
            if mm_block == 0:
                block_beg = 0
            else:
                block_beg = summed_triplet_offsets[mm_block - 1]
            block_end = summed_triplet_offsets[mm_block]

            if x_end - x_beg > 3 * (block_end - block_beg):
                return True, block_beg, block_end

    return False, x_beg, x_end


@wp.kernel(enable_backward=False)
def _bsr_mm_compute_values(
    alpha: Any,
    x_offsets: wp.array(dtype=int),
    x_row_ends: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    x_values: wp.array(dtype=Any),
    y_offsets: wp.array(dtype=int),
    y_row_ends: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    y_values: wp.array(dtype=Any),
    mm_row_min: wp.array(dtype=int),
    summed_triplet_offsets: wp.array(dtype=int),
    summed_triplet_src_blocks: wp.indexedarray(dtype=int),
    mm_row_count: int,
    mm_offsets: wp.array(dtype=int),
    mm_row_ends: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_values: wp.array(dtype=Any),
):
    mm_block = wp.tid()

    row = _bsr_row_index_active(mm_offsets, mm_row_count, mm_block, mm_row_ends)
    if row == -1:
        return

    use_triplets, block_beg, block_end = _bsr_mm_use_triplets(
        row, mm_block, mm_row_min, x_offsets, x_row_ends, summed_triplet_offsets
    )

    mm_val = mm_values.dtype(type(alpha)(0.0))
    col = mm_cols[mm_block]
    if use_triplets:
        for tpl_idx in range(block_beg, block_end):
            x_block = summed_triplet_src_blocks[tpl_idx]
            x_col = x_columns[x_block]
            if x_block != -1:
                y_block = _bsr_block_index_active(x_col, col, y_offsets, y_columns, y_row_ends)
                mm_val += x_values[x_block] * y_values[y_block]
    else:
        for x_block in range(block_beg, block_end):
            x_col = x_columns[x_block]
            y_block = _bsr_block_index_active(x_col, col, y_offsets, y_columns, y_row_ends)
            if y_block != -1:
                mm_val += x_values[x_block] * y_values[y_block]

    mm_values[mm_block] += alpha * mm_val


def make_bsr_mm_compute_values_tiled_outer(subblock_rows, subblock_cols, block_depth, scalar_type, tile_size):
    from warp._src.fem.cache import dynamic_func, dynamic_kernel  # noqa: PLC0415

    mm_type = wp.types.matrix(dtype=scalar_type, shape=(subblock_rows, subblock_cols))

    x_col_vec_t = wp.types.vector(dtype=scalar_type, length=subblock_rows)
    y_row_vec_t = wp.types.vector(dtype=scalar_type, length=subblock_cols)

    suffix = (subblock_rows, subblock_cols, block_depth, tile_size, scalar_type.__name__)

    @dynamic_func(suffix=suffix)
    def _outer_product(
        x_values: wp.array2d(dtype=Any),
        y_values: wp.array2d(dtype=Any),
        brow_off: int,
        bcol_off: int,
        block_col: int,
        brow_count: int,
        bcol_count: int,
    ):
        x_col_vec = x_col_vec_t()
        y_row_vec = y_row_vec_t()

        for k in range(brow_count):
            x_col_vec[k] = x_values[brow_off + k, block_col]
        for k in range(bcol_count):
            y_row_vec[k] = y_values[block_col, bcol_off + k]

        return wp.outer(x_col_vec, y_row_vec)

    @dynamic_kernel(suffix=suffix, kernel_options={"enable_backward": False})
    def bsr_mm_compute_values(
        alpha: Any,
        x_offsets: wp.array(dtype=int),
        x_row_ends: wp.array(dtype=int),
        x_columns: wp.array(dtype=int),
        x_values: wp.array3d(dtype=Any),
        y_offsets: wp.array(dtype=int),
        y_row_ends: wp.array(dtype=int),
        y_columns: wp.array(dtype=int),
        y_values: wp.array3d(dtype=Any),
        mm_row_min: wp.array(dtype=int),
        summed_triplet_offsets: wp.array(dtype=int),
        summed_triplet_src_blocks: wp.indexedarray(dtype=int),
        mm_row_count: int,
        mm_offsets: wp.array(dtype=int),
        mm_row_ends: wp.array(dtype=int),
        mm_cols: wp.array(dtype=int),
        mm_values: wp.array3d(dtype=Any),
    ):
        mm_block, subrow, subcol, lane = wp.tid()

        brow_off = subrow * wp.static(subblock_rows)
        bcol_off = subcol * wp.static(subblock_cols)

        brow_count = wp.min(mm_values.shape[1] - brow_off, subblock_rows)
        bcol_count = wp.min(mm_values.shape[2] - bcol_off, subblock_cols)

        mm_row = _bsr_row_index_active(mm_offsets, mm_row_count, mm_block, mm_row_ends)
        if mm_row == -1:
            return

        lane_val = mm_type()

        use_triplets, block_beg, block_end = _bsr_mm_use_triplets(
            mm_row, mm_block, mm_row_min, x_offsets, x_row_ends, summed_triplet_offsets
        )

        col_count = (block_end - block_beg) * block_depth

        mm_col = mm_cols[mm_block]
        if use_triplets:
            for col in range(lane, col_count, tile_size):
                tpl_block = col // wp.static(block_depth)
                block_col = col - tpl_block * wp.static(block_depth)
                tpl_block += block_beg

                x_block = summed_triplet_src_blocks[tpl_block]
                if x_block != -1:
                    x_col = x_columns[x_block]
                    y_block = _bsr_block_index_active(x_col, mm_col, y_offsets, y_columns, y_row_ends)
                    lane_val += _outer_product(
                        x_values[x_block], y_values[y_block], brow_off, bcol_off, block_col, brow_count, bcol_count
                    )
        else:
            for col in range(lane, col_count, tile_size):
                x_block = col // wp.static(block_depth)
                block_col = col - x_block * wp.static(block_depth)
                x_block += block_beg

                x_col = x_columns[x_block]
                y_block = _bsr_block_index_active(x_col, mm_col, y_offsets, y_columns, y_row_ends)

                if y_block != -1:
                    lane_val += _outer_product(
                        x_values[x_block], y_values[y_block], brow_off, bcol_off, block_col, brow_count, bcol_count
                    )

        mm_val = wp.tile_sum(wp.tile(lane_val, preserve_type=True))[0]

        for coef in range(lane, wp.static(subblock_cols * subblock_rows), tile_size):
            br = coef // subblock_cols
            bc = coef - br * subblock_cols
            if br < brow_count and bc < bcol_count:
                mm_values[mm_block, br + brow_off, bc + bcol_off] += mm_val[br, bc] * alpha

    return bsr_mm_compute_values


@wp.kernel(enable_backward=False)
def _bsr_mm_padded_count(
    beta_nonzero: bool,
    row_count: int,
    col_count: int,
    x_offsets: wp.array(dtype=int),
    x_row_ends: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_row_ends: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    z_offsets: wp.array(dtype=int),
    z_row_ends: wp.array(dtype=int),
    z_columns: wp.array(dtype=int),
    row_block_counts: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    if row >= row_count:
        return

    previous_col = int(-1)
    block_count = int(0)
    searching = bool(True)

    while searching:
        next_col = col_count

        if beta_nonzero:
            for z_block in range(z_offsets[row], z_row_ends[row]):
                col = z_columns[z_block]
                if col > previous_col and col < next_col:
                    next_col = col

        for x_block in range(x_offsets[row], x_row_ends[row]):
            x_col = x_columns[x_block]
            for y_block in range(y_offsets[x_col], y_row_ends[x_col]):
                col = y_columns[y_block]
                if col > previous_col and col < next_col:
                    next_col = col

        if next_col == col_count:
            searching = False
        else:
            block_count += 1
            previous_col = next_col

    if z_offsets[row] + block_count > z_offsets[row + 1]:
        row_block_counts[row] = -1
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
    else:
        row_block_counts[row] = block_count


def make_bsr_mm_padded_fill(block_depth: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=block_depth, kernel_options={"enable_backward": False})
    def bsr_mm_padded_fill(
        alpha: Any,
        beta: Any,
        beta_nonzero: bool,
        row_count: int,
        col_count: int,
        x_offsets: wp.array(dtype=int),
        x_row_ends: wp.array(dtype=int),
        x_columns: wp.array(dtype=int),
        x_values: wp.array3d(dtype=Any),
        y_offsets: wp.array(dtype=int),
        y_row_ends: wp.array(dtype=int),
        y_columns: wp.array(dtype=int),
        y_values: wp.array3d(dtype=Any),
        z_offsets: wp.array(dtype=int),
        old_z_row_ends: wp.array(dtype=int),
        old_z_columns: wp.array(dtype=int),
        old_z_values: wp.array3d(dtype=Any),
        z_row_ends: wp.array(dtype=int),
        z_columns: wp.array(dtype=int),
        z_values: wp.array3d(dtype=Any),
        row_block_counts: wp.array(dtype=int),
    ):
        row, br, bc = wp.tid()

        if row >= row_count:
            return

        row_beg = z_offsets[row]
        capacity_end = z_offsets[row + 1]
        block_count = row_block_counts[row]

        if block_count < 0:
            if br == 0 and bc == 0:
                z_row_ends[row] = row_beg
                for block in range(row_beg, capacity_end):
                    z_columns[block] = -1
            return

        row_end = row_beg + block_count
        previous_col = int(-1)

        for z_block in range(row_beg, row_end):
            next_col = col_count

            if beta_nonzero:
                for old_z_block in range(z_offsets[row], old_z_row_ends[row]):
                    col = old_z_columns[old_z_block]
                    if col > previous_col and col < next_col:
                        next_col = col

            for x_block in range(x_offsets[row], x_row_ends[row]):
                x_col = x_columns[x_block]
                for y_block in range(y_offsets[x_col], y_row_ends[x_col]):
                    col = y_columns[y_block]
                    if col > previous_col and col < next_col:
                        next_col = col

            value = z_values.dtype(type(alpha)(0.0))

            if beta_nonzero:
                old_z_block = _bsr_block_index_active(row, next_col, z_offsets, old_z_columns, old_z_row_ends)
                if old_z_block != -1:
                    value += beta * old_z_values[old_z_block, br, bc]

            for x_block in range(x_offsets[row], x_row_ends[row]):
                x_col = x_columns[x_block]
                y_block = _bsr_block_index_active(x_col, next_col, y_offsets, y_columns, y_row_ends)
                if y_block != -1:
                    product = z_values.dtype(0.0)
                    for k in range(wp.static(block_depth)):
                        product += x_values[x_block, br, k] * y_values[y_block, k, bc]
                    value += alpha * product

            if br == 0 and bc == 0:
                z_columns[z_block] = next_col
            z_values[z_block, br, bc] = value
            previous_col = next_col

        if br == 0 and bc == 0:
            z_row_ends[row] = row_end
            for block in range(row_end, capacity_end):
                z_columns[block] = -1

    return bsr_mm_padded_fill


class bsr_mm_work_arrays(_BsrStatusMixin):
    """Opaque structure for persisting :func:`bsr_mm` temporary work buffers across calls."""

    def __init__(self):
        self._reset(None)

    def _reset(self, device):
        self.device = device
        self._reset_status()
        self._mm_row_min = None
        self._mm_block_counts = None
        self._mm_rows = None
        self._mm_cols = None
        self._mm_src_blocks = None
        self._old_z_values = None
        self._old_z_offsets = None
        self._old_z_row_ends = None
        self._old_z_columns = None
        self._mm_nnz = 0

    def _allocate_stage_1(self, device, x_nnz: int, z: BsrMatrix, beta: float, z_aliasing: bool):
        if self.device != device:
            self._reset(device)

        # Allocations that do not depend on any computation
        self._copied_z_nnz = z.nnz if beta != 0.0 or z_aliasing else 0

        if self._mm_row_min is None or self._mm_block_counts.size < z.nrow + 1:
            self._mm_row_min = wp.empty(shape=(z.nrow + 1,), dtype=int, device=self.device)
        if self._mm_block_counts is None or self._mm_block_counts.size < x_nnz + 1:
            self._mm_block_counts = wp.empty(shape=(x_nnz + 1,), dtype=int, device=self.device)

        if self._copied_z_nnz > 0:
            if self._old_z_values is None or self._old_z_values.size < self._copied_z_nnz:
                self._old_z_values = wp.empty(shape=(self._copied_z_nnz,), dtype=z.values.dtype, device=self.device)

        if z_aliasing:
            if self._old_z_columns is None or self._old_z_columns.size < z.nnz:
                self._old_z_columns = wp.empty(shape=(z.nnz,), dtype=z.columns.dtype, device=self.device)
            if self._old_z_offsets is None or self._old_z_offsets.size < z.nrow + 1:
                self._old_z_offsets = wp.empty(shape=(z.nrow + 1,), dtype=z.offsets.dtype, device=self.device)
            if self._old_z_row_ends is None or self._old_z_row_ends.size < z.nrow:
                self._old_z_row_ends = wp.empty(shape=(z.nrow,), dtype=z.row_ends.dtype, device=self.device)

    def _allocate_stage_2(self, mm_nnz: int):
        # Allocations that depend on unmerged nnz estimate
        self._mm_nnz = mm_nnz
        if self._mm_rows is None or self._mm_rows.size < mm_nnz:
            self._mm_rows = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)
        if self._mm_cols is None or self._mm_cols.size < mm_nnz:
            self._mm_cols = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)
        if self._mm_src_blocks is None or self._mm_src_blocks.size < mm_nnz:
            self._mm_src_blocks = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)


def bsr_mm(
    x: BsrMatrixOrExpression[BlockType[Rows, Any, Scalar]],
    y: BsrMatrixOrExpression[BlockType[Any, Cols, Scalar]],
    z: BsrMatrix[BlockType[Rows, Cols, Scalar]] | None = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    masked: bool = False,
    work_arrays: bsr_mm_work_arrays | None = None,
    reuse_topology: bool = False,
    tile_size: int = 0,
    max_new_nnz: int | None = None,
    topology: str | None = None,
    overflow: str = "error",
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """Perform the sparse matrix-matrix multiplication ``z := alpha * x @ y + beta * z`` on BSR matrices ``x``, ``y`` and ``z``, and return ``z``.

    The ``x``, ``y`` and ``z`` matrices are allowed to alias.
    If the matrix ``z`` is not provided as input, it will be allocated and treated as zero.

    This method can be graph-captured if either:
     - ``masked=True``
     - ``reuse_topology=True``
     - ``max_new_nnz`` is provided
     - ``topology="padded"`` is used with ``overflow="ignore"`` and supplied
       ``work_arrays``

    Args:
        x: Read-only left operand of the matrix-matrix product.
        y: Read-only right operand of the matrix-matrix product.
        z: Mutable affine operand and result matrix. If ``z`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for the ``x @ y`` product
        beta: Uniform scaling factor for ``z``
        masked: If ``True``, keep the non-zero topology of ``z`` unchanged.
        work_arrays: In most cases, this function will require the use of temporary storage.
          This storage can be reused across calls by passing an instance of
          :class:`bsr_mm_work_arrays` in ``work_arrays``.
        reuse_topology: If ``True``, reuse the product topology information
          stored in ``work_arrays`` rather than recompute it from scratch.
          The matrices ``x``, ``y`` and ``z`` must be structurally similar to
          the previous call in which ``work_arrays`` were populated.
        max_new_nnz: If provided, the maximum number of non-zeros for the matrix-matrix product result
           (not counting the existing non-zeros in ``z``).
        tile_size: If a positive integer, use tiles of this size to compute the matrix-matrix product.
          If negative, disable tile-based computation. Defaults to ``0``, which determines whether to
          use tiles using using an heuristic based on the matrix shape and number of non-zeros..
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          ``masked=True``, ``"cached"`` is equivalent to
          ``reuse_topology=True``, and ``"padded"`` writes the result topology
          into existing destination row capacity.
        overflow: Overflow policy for ``topology="padded"``. ``"error"``
          raises on insufficient row capacity, while ``"ignore"`` records
          status in supplied ``work_arrays`` and leaves overflowing rows
          undefined.
    """

    x, x_scale = _extract_matrix_and_scale(x)
    alpha *= x_scale
    y, y_scale = _extract_matrix_and_scale(y)
    alpha *= y_scale

    if topology is None:
        topology = "masked" if masked else "cached" if reuse_topology else "compact"
    elif topology not in ("compact", "masked", "cached", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if overflow not in ("error", "ignore"):
        raise NotImplementedError("Only overflow='error' and overflow='ignore' are currently implemented")

    if topology == "masked":
        masked = True
    elif topology == "cached":
        reuse_topology = True
    elif topology == "padded" and reuse_topology:
        raise ValueError("reuse_topology is not supported with topology='padded'")

    if z is None:
        if masked or topology == "padded":
            raise ValueError("Left-hand-side 'z' matrix must be provided for this topology policy")

        # If not output matrix is provided, allocate it for convenience
        z_block_shape = (x.block_shape[0], y.block_shape[1])
        if z_block_shape == (1, 1):
            z_block_type = x.scalar_type
        else:
            z_block_type = wp.types.matrix(shape=z_block_shape, dtype=x.scalar_type)
        z = bsr_zeros(x.nrow, y.ncol, block_type=z_block_type, device=x.values.device)
        z.values.requires_grad = x.requires_grad or y.requires_grad
        beta = 0.0

    if x.values.device != y.values.device or x.values.device != z.values.device:
        raise ValueError(
            f"All arguments must reside on the same device, got {x.values.device}, {y.values.device} and {z.values.device}"
        )

    if x.scalar_type != y.scalar_type or x.scalar_type != z.scalar_type:
        raise ValueError(
            f"Matrices must have the same scalar type, got {x.scalar_type}, {y.scalar_type} and {z.scalar_type}"
        )

    if (
        x.block_shape[0] != z.block_shape[0]
        or y.block_shape[1] != z.block_shape[1]
        or x.block_shape[1] != y.block_shape[0]
    ):
        raise ValueError(
            f"Incompatible block sizes for matrix multiplication, got ({x.block_shape}, {y.block_shape}) and ({z.block_shape})"
        )

    if x.nrow != z.nrow or z.ncol != y.ncol or x.ncol != y.nrow:
        raise ValueError(
            f"Incompatible number of rows/columns for matrix multiplication, got ({x.nrow}, {x.ncol}) and ({y.nrow}, {y.ncol})"
        )

    device = z.values.device

    if topology == "padded":
        if overflow == "ignore" and work_arrays is None:
            raise ValueError("`work_arrays` must be supplied when using overflow='ignore'")

        if alpha == 0.0 or x.nnz == 0 or y.nnz == 0:
            return bsr_scale(z, beta)

        if work_arrays is None:
            work_arrays = bsr_mm_work_arrays()

        if not isinstance(alpha, z.scalar_type):
            alpha = z.scalar_type(alpha)
        if not isinstance(beta, z.scalar_type):
            beta = z.scalar_type(beta)

        beta_nonzero = beta != z.scalar_type(0.0)
        x_aliasing = z == x
        y_aliasing = z == y
        z_aliasing = x_aliasing or y_aliasing
        snapshot_z = beta_nonzero or z_aliasing

        work_arrays._allocate_stage_1(device, x.nnz, z, beta if beta_nonzero else 0.0, snapshot_z)
        row_block_counts = work_arrays._mm_row_min
        status = work_arrays._ensure_status(z.device)

        if snapshot_z:
            wp.copy(dest=work_arrays._old_z_row_ends, src=z.row_ends, count=z.nrow)
            wp.copy(dest=work_arrays._old_z_columns, src=z.columns, count=z.nnz)
            wp.copy(dest=work_arrays._old_z_values, src=z.values, count=z.nnz)
            old_z_row_ends = work_arrays._old_z_row_ends
            old_z_columns = work_arrays._old_z_columns
            old_z_values = work_arrays._old_z_values
        else:
            old_z_row_ends = z.row_ends
            old_z_columns = z.columns
            old_z_values = z.values

        x_row_ends = old_z_row_ends if x_aliasing else x.row_ends
        x_columns = old_z_columns if x_aliasing else x.columns
        x_values = _as_3d_array(old_z_values, x.block_shape) if x_aliasing else x.scalar_values

        y_row_ends = old_z_row_ends if y_aliasing else y.row_ends
        y_columns = old_z_columns if y_aliasing else y.columns
        y_values = _as_3d_array(old_z_values, y.block_shape) if y_aliasing else y.scalar_values

        wp.launch(
            _bsr_mm_padded_count,
            dim=z.nrow,
            device=device,
            inputs=[
                beta_nonzero,
                z.nrow,
                z.ncol,
                x.offsets,
                x_row_ends,
                x_columns,
                y.offsets,
                y_row_ends,
                y_columns,
                z.offsets,
                z.row_ends,
                z.columns,
                row_block_counts,
                status,
            ],
        )

        if overflow == "error":
            _bsr_raise_if_status_error(status)

        wp.launch(
            make_bsr_mm_padded_fill(x.block_shape[1]),
            dim=(z.nrow, *z.block_shape),
            device=device,
            inputs=[
                alpha,
                beta,
                beta_nonzero,
                z.nrow,
                z.ncol,
                x.offsets,
                x_row_ends,
                x_columns,
                x_values,
                y.offsets,
                y_row_ends,
                y_columns,
                y_values,
                z.offsets,
                old_z_row_ends,
                old_z_columns,
                _as_3d_array(old_z_values, z.block_shape),
                z.row_ends,
                z.columns,
                z.scalar_values,
                row_block_counts,
            ],
        )
        return z

    if alpha == 0.0 or x.nnz == 0 or y.nnz == 0:
        # Easy case
        return bsr_scale(z, beta)

    z_aliasing = z == x or z == y

    if masked:
        # no need to copy z, scale in-place
        copied_z_nnz = 0
        mm_nnz = z.nnz

        if z_aliasing:
            raise ValueError("`masked=True` is not supported for aliased inputs")

        if beta == 0.0:
            # do not bsr_scale(0), this would not preserve topology
            z.values.zero_()
        else:
            bsr_scale(z, beta)
    elif reuse_topology:
        if work_arrays is None:
            raise ValueError("`work_arrays` must not be ``None`` in order to reuse matrix-matrix product topology")

        copied_z_nnz = work_arrays._copied_z_nnz
        mm_nnz = work_arrays._mm_nnz
    else:
        if work_arrays is None:
            work_arrays = bsr_mm_work_arrays()

        if max_new_nnz is None:
            if device.is_capturing:
                raise RuntimeError(
                    "`bsr_mm` requires either `reuse_topology=True`, `masked=True` or `max_new_nnz` to be set for use in graph capture"
                )
            z.nnz_sync()

        work_arrays._allocate_stage_1(device, x.nnz, z, beta, z_aliasing)
        copied_z_nnz = work_arrays._copied_z_nnz

        # Prefix sum of number of (unmerged) mm blocks per row
        # Use either a thread or a block per row depending on avg nnz/row
        work_arrays._mm_block_counts.zero_()
        count_tile_size = 32
        if not device.is_cuda or x.nnz < 3 * count_tile_size * x.nrow:
            count_tile_size = 1

        wp.launch(
            kernel=make_bsr_mm_count_coeffs(count_tile_size),
            device=device,
            dim=(z.nrow, count_tile_size),
            block_dim=count_tile_size if count_tile_size > 1 else 256,
            inputs=[
                y.ncol,
                copied_z_nnz,
                x.offsets,
                x.row_ends,
                x.columns,
                y.offsets,
                y.row_ends,
                y.columns,
                work_arrays._mm_row_min,
                work_arrays._mm_block_counts,
            ],
        )
        warp._src.utils.array_scan(work_arrays._mm_block_counts[: x.nnz + 1], work_arrays._mm_block_counts[: x.nnz + 1])

        if max_new_nnz is not None:
            mm_nnz = max_new_nnz + copied_z_nnz
        else:
            # Get back total counts on host -- we need a synchronization here
            # Use pinned buffer from z, we are going to need it later anyway
            nnz_buf, _ = z._setup_nnz_transfer()
            stream = wp.get_stream(device) if device.is_cuda else None
            wp.copy(dest=nnz_buf, src=work_arrays._mm_block_counts, src_offset=x.nnz, count=1, stream=stream)
            if device.is_cuda:
                wp.synchronize_stream(stream)
            mm_nnz = int(nnz_buf.numpy()[0])

            if mm_nnz == copied_z_nnz:
                # x@y = 0
                return bsr_scale(z, beta)

        work_arrays._allocate_stage_2(mm_nnz)

        # If z has a non-zero scale, save current data before overwriting it
        if copied_z_nnz > 0:
            # Copy z row and column indices
            wp.copy(dest=work_arrays._mm_cols, src=z.columns, count=copied_z_nnz)
            z.uncompress_rows(out=work_arrays._mm_rows)
            work_arrays._mm_src_blocks[:copied_z_nnz].fill_(-1)
            if z_aliasing:
                # If z is aliasing with x or y, need to save topology as well
                wp.copy(src=z.columns, dest=work_arrays._old_z_columns, count=copied_z_nnz)
                wp.copy(src=z.offsets, dest=work_arrays._old_z_offsets, count=z.nrow + 1)
                wp.copy(src=z.row_ends, dest=work_arrays._old_z_row_ends, count=z.nrow)

        # Fill unmerged mm blocks rows and columns
        wp.launch(
            kernel=_bsr_mm_list_coeffs,
            device=device,
            dim=mm_nnz - copied_z_nnz,
            inputs=[
                copied_z_nnz,
                mm_nnz,
                x.nrow,
                x.offsets,
                x.row_ends,
                x.columns,
                y.offsets,
                y.row_ends,
                y.columns,
                work_arrays._mm_row_min,
                work_arrays._mm_block_counts,
                work_arrays._mm_rows,
                work_arrays._mm_cols,
                work_arrays._mm_src_blocks,
            ],
        )

    alpha = z.scalar_type(alpha)
    beta = z.scalar_type(beta)

    if copied_z_nnz > 0:
        # Save current z values in temporary buffer
        wp.copy(src=z.values, dest=work_arrays._old_z_values, count=copied_z_nnz)

    if not masked:
        # Increase dest array size if needed
        if z.columns.shape[0] < mm_nnz:
            z.columns = wp.empty(shape=(mm_nnz,), dtype=int, device=device)

        from warp._src.context import runtime  # noqa: PLC0415

        if device.is_cpu:
            native_func = runtime.core.wp_bsr_matrix_from_triplets_host
        else:
            native_func = runtime.core.wp_bsr_matrix_from_triplets_device

        nnz_buf, nnz_event = z._setup_nnz_transfer()
        summed_triplet_offsets = wp.empty(shape=(mm_nnz,), dtype=wp.int32, device=device)
        summed_triplet_indices = wp.empty(shape=(mm_nnz,), dtype=wp.int32, device=device)

        with wp.ScopedDevice(z.device):
            native_func(
                z.block_size,
                0,  # scalar_size_in_bytes
                z.nrow,
                z.ncol,
                mm_nnz,
                None,  # device nnz
                ctypes.cast(work_arrays._mm_rows.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(work_arrays._mm_cols.ptr, ctypes.POINTER(ctypes.c_int32)),
                None,  # triplet values
                0,  # zero_value_mask
                False,  # masked_topology
                ctypes.cast(summed_triplet_offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(summed_triplet_indices.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(z.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(z.row_ends.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(z.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
                _optional_ctypes_event(nnz_event),
            )
        _bsr_set_compact_row_ends(z)

        # Resize z to fit mm result if necessary
        # If we are not reusing the product topology, this needs another synchronization
        if not reuse_topology:
            work_arrays.result_nnz = z.nnz_sync() if max_new_nnz is None else mm_nnz

        _bsr_ensure_fits(z, nnz=work_arrays.result_nnz)
        z.values.zero_()

        if copied_z_nnz > 0:
            # Add back original z values
            wp.launch(
                kernel=_bsr_axpy_add_block,
                device=device,
                dim=(copied_z_nnz, z.block_shape[0], z.block_shape[1]),
                inputs=[
                    0,
                    beta,
                    work_arrays._mm_rows,
                    work_arrays._mm_cols,
                    z.offsets,
                    z.row_ends,
                    z.columns,
                    _as_3d_array(work_arrays._old_z_values, z.block_shape),
                    z.scalar_values,
                ],
            )

    max_subblock_dim = 12
    if tile_size > 0:
        use_tiles = True
    elif tile_size < 0:
        use_tiles = False
    else:
        # Heuristic for using tiled variant: few or very large blocks
        tile_size = 64
        max_tiles_per_sm = 2048 // tile_size  # assume 64 resident warps per SM
        use_tiles = device.is_cuda and (
            max(x.block_size, y.block_size, z.block_size) > max_subblock_dim**2
            or z.nnz < max_tiles_per_sm * device.sm_count
        )

    if use_tiles:
        subblock_rows = min(max_subblock_dim, z.block_shape[0])
        subblock_cols = min(max_subblock_dim, z.block_shape[1])

        wp.launch(
            kernel=make_bsr_mm_compute_values_tiled_outer(
                subblock_rows, subblock_cols, x.block_shape[1], z.scalar_type, tile_size
            ),
            device=device,
            dim=(
                z.nnz,
                (z.block_shape[0] + subblock_rows - 1) // subblock_rows,
                (z.block_shape[1] + subblock_cols - 1) // subblock_cols,
                tile_size,
            ),
            block_dim=tile_size,
            inputs=[
                alpha,
                work_arrays._old_z_offsets if x == z else x.offsets,
                work_arrays._old_z_row_ends if x == z else x.row_ends,
                work_arrays._old_z_columns if x == z else x.columns,
                _as_3d_array(work_arrays._old_z_values, z.block_shape) if x == z else x.scalar_values,
                work_arrays._old_z_offsets if y == z else y.offsets,
                work_arrays._old_z_row_ends if y == z else y.row_ends,
                work_arrays._old_z_columns if y == z else y.columns,
                _as_3d_array(work_arrays._old_z_values, z.block_shape) if y == z else y.scalar_values,
                None if masked else work_arrays._mm_row_min,
                None if masked else summed_triplet_offsets,
                None if masked else work_arrays._mm_src_blocks[summed_triplet_indices],
                z.nrow,
                z.offsets,
                z.row_ends,
                z.columns,
                z.scalar_values,
            ],
        )

        return z

    # Add mm blocks to z values
    if (type_is_matrix(x.values.dtype) or type_is_matrix(y.values.dtype)) and not (type_is_matrix(z.values.dtype)):
        # Result block type is scalar, but operands are matrices
        # Cast result to (1x1) matrix to perform multiplication
        mm_values = z.values.view(wp.types.matrix(shape=(1, 1), dtype=z.scalar_type))
    else:
        mm_values = z.values

    wp.launch(
        kernel=_bsr_mm_compute_values,
        device=device,
        dim=z.nnz,
        inputs=[
            alpha,
            work_arrays._old_z_offsets if x == z else x.offsets,
            work_arrays._old_z_row_ends if x == z else x.row_ends,
            work_arrays._old_z_columns if x == z else x.columns,
            work_arrays._old_z_values if x == z else x.values,
            work_arrays._old_z_offsets if y == z else y.offsets,
            work_arrays._old_z_row_ends if y == z else y.row_ends,
            work_arrays._old_z_columns if y == z else y.columns,
            work_arrays._old_z_values if y == z else y.values,
            None if masked else work_arrays._mm_row_min,
            None if masked else summed_triplet_offsets,
            None if masked else work_arrays._mm_src_blocks[summed_triplet_indices],
            z.nrow,
            z.offsets,
            z.row_ends,
            z.columns,
            mm_values,
        ],
    )

    return z


def make_bsr_mv_kernel(block_cols: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=block_cols, kernel_options={"enable_backward": False})
    def bsr_mv_kernel(
        alpha: Any,
        A_offsets: wp.array(dtype=int),
        A_row_ends: wp.array(dtype=int),
        A_columns: wp.array(dtype=int),
        A_values: wp.array3d(dtype=Any),
        x: wp.array(dtype=Any),
        beta: Any,
        y: wp.array(dtype=Any),
    ):
        row, subrow = wp.tid()

        block_rows = A_values.shape[1]

        yi = row * block_rows + subrow

        # zero-initialize with type of y elements
        scalar_zero = type(alpha)(0)
        v = scalar_zero

        if alpha != scalar_zero:
            beg = A_offsets[row]
            end = A_row_ends[row]
            for block in range(beg, end):
                xs = A_columns[block] * block_cols
                for col in range(wp.static(block_cols)):
                    v += A_values[block, subrow, col] * x[xs + col]
            v *= alpha

        if beta != scalar_zero:
            v += beta * y[yi]

        y[yi] = v

    return bsr_mv_kernel


def make_bsr_mv_tiled_kernel(tile_size: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=tile_size, kernel_options={"enable_backward": False})
    def bsr_mv_tiled_kernel(
        alpha: Any,
        A_offsets: wp.array(dtype=int),
        A_row_ends: wp.array(dtype=int),
        A_columns: wp.array(dtype=int),
        A_values: wp.array3d(dtype=Any),
        x: wp.array(dtype=Any),
        beta: Any,
        y: wp.array(dtype=Any),
    ):
        row, subrow, lane = wp.tid()

        scalar_zero = type(alpha)(0)
        block_rows = A_values.shape[1]
        block_cols = A_values.shape[2]

        yi = row * block_rows + subrow

        if beta == scalar_zero:
            subrow_sum = wp.tile_zeros(shape=(1,), dtype=y.dtype)
        else:
            subrow_sum = beta * wp.tile_load(y, 1, yi)

        if alpha != scalar_zero:
            block_beg = A_offsets[row]
            col_count = (A_row_ends[row] - block_beg) * block_cols

            col = lane
            lane_sum = y.dtype(0)

            for col in range(lane, col_count, tile_size):
                block = col // block_cols
                block_col = col - block * block_cols
                block += block_beg

                xi = x[A_columns[block] * block_cols + block_col]
                lane_sum += A_values[block, subrow, block_col] * xi

            lane_sum *= alpha
            subrow_sum += wp.tile_sum(wp.tile(lane_sum))

        wp.tile_store(y, subrow_sum, yi)

    return bsr_mv_tiled_kernel


def make_bsr_mv_transpose_kernel(block_rows: int):
    from warp._src.fem.cache import dynamic_kernel  # noqa: PLC0415

    @dynamic_kernel(suffix=block_rows, kernel_options={"enable_backward": False})
    def bsr_mv_transpose_kernel(
        alpha: Any,
        A_row_count: int,
        A_offsets: wp.array(dtype=int),
        A_row_ends: wp.array(dtype=int),
        A_columns: wp.array(dtype=int),
        A_values: wp.array3d(dtype=Any),
        x: wp.array(dtype=Any),
        y: wp.array(dtype=Any),
    ):
        block, subcol = wp.tid()

        row = _bsr_row_index_active(A_offsets, A_row_count, block, A_row_ends)
        if row == -1:
            return

        block_cols = A_values.shape[2]

        A_block = A_values[block]

        col_sum = type(alpha)(0)
        for subrow in range(wp.static(block_rows)):
            col_sum += A_block[subrow, subcol] * x[row * block_rows + subrow]

        wp.atomic_add(y, A_columns[block] * block_cols + subcol, alpha * col_sum)

    return bsr_mv_transpose_kernel


def _vec_array_view(array: wp.array, dtype: type, expected_scalar_count: int) -> wp.array:
    # cast a 1d or 2d array to a 1d array with the target dtype, adjusting shape as required

    scalar_count = array.size * type_size(array.dtype)
    if scalar_count != expected_scalar_count:
        raise ValueError(f"Invalid array scalar size, expected {expected_scalar_count}, got {scalar_count}")

    if array.ndim == 1 and types_equal(array.dtype, dtype):
        return array

    if type_scalar_type(array.dtype) != type_scalar_type(dtype):
        raise ValueError(f"Incompatible scalar types, expected {type_repr(array.dtype)}, got {type_repr(dtype)}")

    if array.ndim > 2:
        raise ValueError(f"Incompatible array number of dimensions, expected 1 or 2, got {array.ndim}")

    if not array.is_contiguous:
        raise ValueError("Array must be contiguous")

    vec_length = type_size(dtype)
    vec_count = scalar_count // vec_length
    if vec_count * vec_length != scalar_count:
        raise ValueError(
            f"Array of shape {array.shape} and type {type_repr(array.dtype)} cannot be reshaped to an array of type {type_repr(dtype)}"
        )

    def vec_view(array):
        return wp.array(
            data=None,
            ptr=array.ptr,
            capacity=array.capacity,
            device=array.device,
            dtype=dtype,
            shape=vec_count,
            grad=None if array.grad is None else vec_view(array.grad),
        )

    view = vec_view(array)
    view._ref = array
    return view


def bsr_mv(
    A: BsrMatrixOrExpression[BlockType[Rows, Cols, Scalar]],
    x: Array[Vector[Scalar, Cols] | Scalar],
    y: Array[Vector[Scalar, Rows] | Scalar] | None = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    transpose: bool = False,
    work_buffer: Array[Vector[Scalar, Rows] | Scalar] | None = None,
    tile_size: int = 0,
) -> Array[Vector[Scalar, Rows] | Scalar]:
    """Perform the sparse matrix-vector product ``y := alpha * A * x + beta * y`` and return ``y``.

    The ``x`` and ``y`` vectors are allowed to alias.

    Args:
        A: Read-only, left matrix operand of the matrix-vector product.
        x: Read-only, right vector operand of the matrix-vector product.
        y: Mutable affine operand and result vector. If ``y`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for ``x``. If zero, ``x`` will not be read and may be left uninitialized.
        beta: Uniform scaling factor for ``y``. If zero, ``y`` will not be read and may be left uninitialized.
        transpose: If ``True``, use the transpose of the matrix ``A``. In this case the result is **non-deterministic**.
        work_buffer: Temporary storage is required if and only if ``x`` and ``y`` are the same vector.
          If provided, the ``work_buffer`` array will be used for this purpose,
          otherwise a temporary allocation will be performed.
        tile_size: If a positive integer, use tiles of this size to compute the matrix-matrix product.
          If negative, disable tile-based computation. Defaults to ``0``, which determines whether to
          use tiles using using an heuristic based on the matrix shape and number of non-zeros..
    """

    A, A_scale = _extract_matrix_and_scale(A)
    alpha *= A_scale

    if transpose:
        block_shape = A.block_shape[1], A.block_shape[0]
        nrow, ncol = A.ncol, A.nrow
    else:
        block_shape = A.block_shape
        nrow, ncol = A.nrow, A.ncol

    if y is None:
        # If no output array is provided, allocate one for convenience
        y_vec_len = block_shape[0]
        y_dtype = A.scalar_type if y_vec_len == 1 else wp.types.vector(length=y_vec_len, dtype=A.scalar_type)
        y = wp.empty(shape=(nrow,), device=A.values.device, dtype=y_dtype, requires_grad=x.requires_grad)
        beta = 0.0

    alpha = A.scalar_type(alpha)
    beta = A.scalar_type(beta)

    device = A.values.device
    if A.values.device != x.device or A.values.device != y.device:
        raise ValueError(
            f"A, x, and y must reside on the same device, got {A.values.device}, {x.device} and {y.device}"
        )

    if x.ptr == y.ptr:
        # Aliasing case, need temporary storage
        if work_buffer is None:
            work_buffer = wp.empty_like(y)
        elif work_buffer.size < y.size:
            raise ValueError(f"Work buffer size is insufficient, needs to be at least {y.size}, got {work_buffer.size}")
        elif not types_equal(work_buffer.dtype, y.dtype):
            raise ValueError(
                f"Work buffer must have same data type as y, {type_repr(y.dtype)} vs {type_repr(work_buffer.dtype)}"
            )

        # Save old y values before overwriting vector
        wp.copy(dest=work_buffer, src=y, count=y.size)
        x = work_buffer

    try:
        x_view = _vec_array_view(x, A.scalar_type, expected_scalar_count=ncol * block_shape[1])
    except ValueError as err:
        raise ValueError("Incompatible 'x' vector for bsr_mv") from err
    try:
        y_view = _vec_array_view(y, A.scalar_type, expected_scalar_count=nrow * block_shape[0])
    except ValueError as err:
        raise ValueError("Incompatible 'y' vector for bsr_mv") from err

    # heuristic to use tiled version for long rows
    if tile_size > 0:
        use_tiles = True
    elif tile_size < 0:
        use_tiles = False
    else:
        tile_size = 64
        use_tiles = device.is_cuda and A.nnz * A.block_size > 2 * tile_size * A.shape[0]

    if transpose:
        if beta.value == 0.0:
            y.zero_()
        elif beta.value != 1.0:
            wp.launch(
                kernel=_bsr_scale_kernel,
                device=y.device,
                dim=y_view.shape[0],
                inputs=[beta, y_view],
            )
        if alpha.value != 0.0:
            wp.launch(
                kernel=make_bsr_mv_transpose_kernel(block_rows=block_shape[1]),
                device=A.values.device,
                dim=(A.nnz, block_shape[0]),
                inputs=[alpha, A.nrow, A.offsets, A.row_ends, A.columns, A.scalar_values, x_view, y_view],
            )
    elif use_tiles:
        wp.launch(
            kernel=make_bsr_mv_tiled_kernel(tile_size),
            device=A.values.device,
            dim=(nrow, block_shape[0], tile_size),
            block_dim=tile_size,
            inputs=[alpha, A.offsets, A.row_ends, A.columns, A.scalar_values, x_view, beta, y_view],
        )
    else:
        wp.launch(
            kernel=make_bsr_mv_kernel(block_cols=block_shape[1]),
            device=A.values.device,
            dim=(nrow, block_shape[0]),
            inputs=[alpha, A.offsets, A.row_ends, A.columns, A.scalar_values, x_view, beta, y_view],
        )

    return y
