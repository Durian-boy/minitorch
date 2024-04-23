from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret


    # 二维的矩阵乘法我们都很熟悉，其实高维的矩阵（张量）乘法也可以被视为多次的二维矩阵乘法
    # 对于(2,2,3)这个张量来说，我们可以把它看作是两个(2,3)的张量A(dim0=0)和B(dim0=1)摞起来
    # 对于(1,3,2)这个张量来说，我们可以把它看作是一个(3,2)的张量C(dim0=0)
    # 所以对于高维的张量来说，只有最后两维表示矩阵，其他维度都只表示矩阵的排列而已
    # 准确来说除了最后两个维度，其他维度的值的变化都代表指向不同的二维矩阵
    # 此时若这两个张量进行乘法，首先要将(1,3,2)进行广播至(2,3,2)即两个C摞列
    # 此时已经可以确定最后输出张量的形状了：(2,2,2)，因为形状(2,3)×(3,2)=(2,2)
    # 对于输出张量dim0=0时的2×2矩阵，是由A×C得来
    # 对于输出张量dim0=0时的2×2矩阵，是由B×C得来
    # 除了最后两维，其他维要符合广播机制才能做矩阵乘法
    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """
        # 用于记录两个输入张量都否都为2维
        both_2d = 0
        # 要确保两个相乘的矩阵的维数>=3,如果为2维矩阵，则扩展为3维矩阵
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        # both==2如果为真，则both_2d = True，反之为False
        both_2d = both_2d == 2

        # 因为除了最后两维的其他维度要符合广播条件，所以取出来存入ls，对这些维度进行广播
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        # 相乘后的矩阵的行应等于a最后两维中的行数，列数应等于b最后两维中的列数
        # 即矩阵规模为a.shape[-2] × b.shape[-1]，此时输出张量的形状已经确定了，就是ls
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        # a最后两维中的列数和b最后两维中的行数相同是矩阵乘法的基本要求
        assert a.shape[-1] == b.shape[-2]
        # 按照ls创建一个零张量out
        out = a.zeros(tuple(ls))

        # 这个函数的逻辑需要我们自己实现
        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # 如果两个输入张量为2维，则需要将结果还原为2维
        # 如果有其中一个张量为3维或以上，则其结果本来就是3维或以上，不用还原
        if both_2d:
            # 将扩展的维度删除
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # 用prange代替range实现并行
        for i in prange(len(out)):
            # 按形状创建数组用来存储索引
            # 这块数组的定义最好放在for循环内，每个线程都有独立的空间来存储索引
            out_index = np.zeros(len(out_shape), np.int32)
            in_index = np.zeros(len(in_shape), np.int32)
            # 将1-D序数转换为多维的索引
            to_index(i, out_shape, out_index)
            # 将输出张量的索引转换为输入张量的索引
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # 将输入张良的索引转换为其1-D的序数，并从in-storage中取出相应元素
            data = in_storage[index_to_position(in_index, in_strides)]
            # 对去除的元素进行映射操作
            map_data = fn(data)
            # 将计算结果存入输出张量的相应位置
            out[index_to_position(out_index, out_strides)] = map_data
    # 使用numba优化进行并行计算
    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # 用prange代替range实现并行
        for i in prange(len(out)):
            # 用形状来创建数组存储索引
            # 这块数组的定义最好放在for循环内，每个线程都有独立的空间来存储索引
            out_index = np.zeros(len(out_shape), np.int32)
            a_in = np.zeros(len(a_shape), np.int32)
            b_in = np.zeros(len(b_shape), np.int32)
            # 将1-D序数i转换为在输出张量中的索引
            to_index(i, out_shape, out_index)
            # 将输出张量的索引转换为a输入张量的索引
            broadcast_index(out_index, out_shape, a_shape, a_in)
            # 将a输入张量中的值取出来
            a = a_storage[index_to_position(a_in, a_strides)]
            # 将输出张量的索引转换为b输入张量的索引
            broadcast_index(out_index, out_shape, b_shape, b_in)
            # 将b输入张量中的值取出来
            b = b_storage[index_to_position(b_in, b_strides)]
            # 进行联合运算后存入输出张量的相应位置
            out[i] = fn(a, b)

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # 用prange代替range实现并行
        for i in range(len(out)):
            # 用形状来创建数组存储索引
            # 这块数组的定义最好放在for循环内，每个线程都有独立的空间来存储索引
            out_index = np.zeros(len(out_shape), np.int32)
            # 将1-D序数i转换为在输出张量中的索引
            to_index(i, out_shape, out_index)
            for j in range(a_shape[reduce_dim]):
                # 此时a_index为刚刚由i转换来的输出张量的索引
                a_index = out_index.copy()
                # 因为输出张量的形状在reduce_dim上的取值为1，所以让这个值为j
                # 以此来遍历输入张量在该dim上的所有元素，并决定是累加还是累乘
                a_index[reduce_dim] = j
                # 将该输入张量的索引转换为其1-D序数
                pos_a = index_to_position(a_index, a_strides)
                # 其实就是一个累加/累乘的过程
                # 将pos_a的元素累加/累乘到out[index]的位置
                # 但是如何保证add时，out的每个元素是0.0？在mul时，out的每个元素是1.0？
                out[i] = fn(a_storage[pos_a], out[i])

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # 目标：遍历并填充out中的每一个位置
    n = len(out)
    for i in prange(n):
        out_index = out_shape.copy()
        temp_i = i + 0
        # 将1-D序数转换为输出张量的索引，存储在out_index中
        to_index(temp_i, out_shape, out_index)
        # a的最后一个维度就是二维矩阵相乘时会被消去的维度：2×3 * 3×2 = 2×2
        # 同时也表示有多少个元素要两两相乘最后求和变为输出张量中的一个元素
        for j in range(a_shape[-1]):
            temp_j = j + 0
            a_index = a_shape.copy()
            a_tmp_index = out_index.copy()
            # 其他维度不变，只改变列坐标，相当于遍历行
            a_tmp_index[-1] = temp_j
            # 将a_tmp_index索引转换为在a中的索引
            broadcast_index(a_tmp_index, out_shape, a_shape, a_index)
            # 将a的索引转换为a的1-D序数，并取出相应元素
            a_pos = index_to_position(a_index, a_strides)

            b_index = b_shape.copy()
            b_tmp_index = out_index.copy()
            # 其他维度不变，只改变行坐标，相当于遍历列
            b_tmp_index[-2] = temp_j
            # 将b_tmp_index索引转换为在b中的索引
            broadcast_index(b_tmp_index, out_shape, b_shape, b_index)
            # 将b的索引转换为b的1-D序数，并取出相应元素
            b_pos = index_to_position(b_index, b_strides)

            # 两元素相乘并累加
            out[temp_i] += (a_storage[a_pos] * b_storage[b_pos])


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
