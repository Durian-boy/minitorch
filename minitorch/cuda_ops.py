from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3-dimensional multiply
        # 用于判断是否是两个二维张量相乘
        both_2d = 0
        # 如果是二维张量，则将其扩展为三维
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        # 对除了最后两维的其他维度进行广播
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        # 再将最后两维附加在最后
        # 相乘后的张量最后两维尺寸应是第一个张量的行数，和第二个张量的列数
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        # 二维矩阵相乘的基本要求
        assert a.shape[-1] == b.shape[-2]
        # 按形状ls生成零张量
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        # block在grid中是三维分布的，因为CUDA网络是一个三维结构
        # 这里使用out的shape[0]定义三维结构中有多少“层”（也是batch数）, shape[1], shape[2]来计算三维结构，行和列的尺寸
        # 其实这里用shape[-2]和shape[-1]来计算可能会更好理解点
        """
        blockspergrid = (
            (out.shape[-2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[-1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (len(out._tensor._storage) + THREADS_PER_BLOCK * THREADS_PER_BLOCK - 1)
            // (THREADS_PER_BLOCK * THREADS_PER_BLOCK),
        )
        """

        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )

        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)   # (32,32,1)

        # 配置CUDA参数，并调用矩阵相乘实际操作的函数
        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # 如果原来是两个二维张量，则将其还原为二维
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # 这里MAX_DIMS=32
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        # 计算线程的全局索引
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # 判断当前线程的索引是否小于输出的元素个数，因为线程的个数可能多于元素个数
        if i < out_size:
            # 剩下的操作和非并行的map一致
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_ordinal = index_to_position(in_index, in_strides)
            out[i] = fn(in_storage[in_ordinal])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # 为每个线程申请本地内存
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            i = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_ordinal = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_ordinal = index_to_position(b_index, b_strides)
            out[i] = fn(a_storage[a_ordinal], b_storage[b_ordinal])

    return cuda.jit()(_zip)  # type: ignore


# 将输入数组按32为长度进行分段，并对每段进行求和
# 每个线程块有32个线程，所以分段累加的结果就是每个线程块中所有线程求和的结果
def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    # 每个线程块包含的线程数
    BLOCK_DIM = 32
    # 分配了一个共享内存数组，用于存储线程块内每个线程读取到的输入元素以及中间累加结果
    shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)
    # 计算当前线程的全局索引，由线程块索引和线程索引共同决定
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # 记录当前线程在所属线程块中的索引，等于线程索引
    local_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x

    # 若该现成的全局索引超出a的长度，则将对应的共享内存位置初始化为0
    if i >= size:
        shared_block[local_idx] = 0
    # 否则将a中对应i位置的元素存入共享内存相应位置
    else:
        shared_block[local_idx] = a[i]
    # 进行线程同步，确保所有线程完成数据加载后再继续执行
    cuda.syncthreads()
    offset = 1

    # 这段程序的运行过程：
    # 第一轮：让每个线程并行的判断：自己的线程id（local_idx）是否为2的整数倍，如果可以，那么就将右偏移1个位置的元素累加到该线程所对应的shared_block的位置
    # 第一轮是以2长度来分段，并对每一段求和放入每段的首元素位置上
    # 第二轮：让每个线程并行的判断：自己的线程id（local_idx）是否为4的整数倍，如果可以，那么就将右偏移2个位置的元素累加到该线程所对应的shared_block的位置
    # 第二轮是以4为长度来分段，并利用上一轮求和的结果，这样以来，只需要求和一次就可得到4个元素的加总
    # 第三轮：让每个线程并行的判断：自己的线程id（local_idx）是否为8的整数倍，如果可以，那么就将右偏移4个位置的元素累加到该线程所对应的shared_block的位置
    # 第三轮同理，获得了每8个元素的加总，并将结果存入每8个元素的首元素位置
    # 第四轮：让每个线程并行的判断：自己的线程id（local_idx）是否为16的整数倍，如果可以，那么就将右偏移8个位置的元素累加到该线程所对应的shared_block的位置
    # 第四轮同理，获得了每16个元素的加总，并将结果存入每16个元素的首元素位置
    # 第五轮：让每个线程并行的判断：自己的线程id（local_idx）是否为32的整数倍，如果可以，那么就将右偏移16个位置的元素累加到该线程所对应的shared_block的位置
    # 第五轮实际上只有shared_block[0]可以通过判断，并将shard_block[16]的值与自己加总，至此，就完成了对32个元素的并行求和
    # 以上过程对元素不满32个时也适用
    while offset < BLOCK_DIM:
        numba.cuda.syncthreads()
        # 注意：0对任何数取余都为0
        if local_idx % (offset * 2) == 0:
            shared_block[local_idx] += shared_block[local_idx + offset]
        offset *= 2

    # 至此已完成每段的求和，但是结果还存储在每不同的shared_block的首位置
    # 每个线程block对应out中的一个位置，按block_index，将每段求和结果存入out
    out[block_idx] = shared_block[0]


# 定义了一个名为jit_sum_practice的装饰器，该装饰器用于将_sum_practice函数转换为编译后的CUDA内核
jit_sum_practice = cuda.jit()(_sum_practice)


# 对_sum_practice的进一步封装
def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK  # 32
    # 因为除法向下取整，还需要加1，我们希望在同一个grid内完成计算
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    # 定义out的storage和shape，strides为默认值None
    out = TensorData([0.0 for i in range(2)], (2,))
    # 张量数据存入CUDA设备
    out.to_cuda_()
    # 配置CUDA内核编译参数：blockspergrid, threadsperblock，传参并调用_sum_practice
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        # 需要被归约维度的尺寸
        reduce_size = a_shape[reduce_dim]
        local_idx = numba.cuda.threadIdx.x
        block_idx = numba.cuda.blockIdx.x
        shared_block = numba.cuda.shared.array(BLOCK_DIM, numba.float64)
        offset = 1
        out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
        # 将每个线程所在的block号转变为o输出张量的索引
        # 输出张量的每一个元素对应一个block
        to_index(block_idx, out_shape, out_index)

        # 当线程编号小于要加总的数组长度时
        if local_idx < reduce_size:
            # 每一个线程对应一个local_idx，其实这一步就相当于是遍历a中要被归约维度的元素
            out_index[reduce_dim] = local_idx
            # 将a的索引转换为1-D序号，从a中取出相应元素，放入shared_block相应位置
            shared_block[local_idx] = a_storage[index_to_position(out_index, a_strides)]

        # 当线程编号超出要加总的数组长度时
        else:
            # 将线程对应的shared_block的值设置为传入的值
            shared_block[local_idx] = reduce_value

        # 对每个block进行并行求和
        while offset < BLOCK_DIM:
            numba.cuda.syncthreads()
            if local_idx % (offset * 2) == 0:
                # 应用fn，并将结果存储
                shared_block[local_idx] = fn(
                    shared_block[local_idx], shared_block[local_idx + offset]
                )
            offset *= 2

        # 同步线程，等待所有线程都到这一步，再进行下一步
        numba.cuda.syncthreads()
        # 将每个block的首元素存入输出张量，完成归约
        if local_idx == 0:
            out[block_idx] = shared_block[local_idx]

    return cuda.jit()(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # 由于a和b都是[size, size]尺寸的，且size恒小于32
    # 所以申请两个个[32, 32]的共享内存，用于存储a和b两个张量
    shared_a = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    y = numba.cuda.threadIdx.y
    x = numba.cuda.threadIdx.x
    # 如果x和y没有越界
    if x < size and y < size:
        # 将相应的值都存入相应的共享内存中
        shared_a[y, x] = a[y * size + x]
        shared_b[y, x] = b[y * size + x]
    else:
        # 对于越界的线程，将其共享内存对应的值设为0
        shared_a[y, x] = 0
        shared_b[y, x] = 0
    # 同步线程
    numba.cuda.syncthreads()

    # 在x和y没有越界情况下
    if y < size and x < size:
        temp = 0
        # 对每个线程来说，计算其应计算的相应行和列的乘积
        for val in range(size):
            # a的第y行与b的第x列相乘
            temp += shared_a[y, val] * shared_b[val, x]
        # 将结果存入out[y, x]
        out[y * size + x] = temp


# 定义了一个名为jit_mm_practice的装饰器，该装饰器用于将_mm_practice函数转换为编译后的CUDA内核
jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)  # (32, 32)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    # 将out存入CUDA设备中，a和b不用存是因为a和b会在jit_mm_practice传参时被存入设备
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # 表示现在在哪一“层”
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    # 共享内存的形状为32*32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # i为输出张量中的元素在x方向的全局索引，j为输出张量中的元素在y方向的全局索引
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # 用于保存累加结果
    acc = 0.0
    # k_offset会从0开始，每次增加BLOCK_DIM=32，直到k_offset>=a_shape[2]退出迭代
    # k_offset + pj实际上是将a的行号固定下来为i后，遍历了所有列号，只是遍历时是一段一段（32长度）遍历的
    # 在遍历每一段时有32个线程并行的将a中这一段第i行的元素取出来存入共享内存的第pi行（共享内存对同一块内的线程来说是共享的）
    # 这32个线程由于处在同一个block，且处于同一行，所以计算出来的i相同，每一次k_offset的迭代，就相当于是这32个线程沿列变化的方向
    # 平移了32个长度，然后再从a的第i行取32个元素存入共享内存的第pi行，对b来说也是同理，不同之处只是遍历的是行（行变化方向）
    # 每一次迭代其实都会清空共享内存，所以有必要要在清空之前将共享内存的元素完成计算并保存下来，
    # 两块共享内存都是32*32的，相乘结果也是32*32，线程也有32*32个，所以每次循环都每个线程除了要完成
    # a和b中各一个元素的获取任务，还需要计算a_shared pi行和b_shared pj列的内积，也就是每个线程完成a_shared和b_shared矩阵相乘后
    # 其中(pi,pj)位置元素的计算，每次循环都将结果累加，直至循环结束，其实这里应用了【分块矩阵乘法】，将大矩阵拆分为小的矩阵然后再用
    # 小矩阵参加矩阵乘法，最终将结果放在结果矩阵中合适的位置，而在这个例子中，小矩阵的尺寸就是32*32
    for k_offset in range(0, a_shape[2], BLOCK_DIM):
        # 记存储a中行索引和b中的列索引
        k = (k_offset + pj, k_offset + pi)

        # 判断(i, k[0])点是否在a的范围内
        if i < a_shape[1] and k[0] < a_shape[2]:
            # 如果在，则计算出其1-D序数
            a_pos = a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k[0]
            # 对a来说，每个线程在每段中，只负责一个元素的获取，从a中取出(i, k[0])存入共享内存a中的(pi, pj)位置
            a_shared[pi, pj] = a_storage[a_pos]

        # 判断(k[1], j)点是否在b的范围内
        if j < b_shape[2] and k[1] < b_shape[1]:
            # 如果在，则计算出其1-D序数
            b_pos = b_batch_stride * batch + b_strides[1] * k[1] + b_strides[2] * j
            # 对b来说，每个线程在每段中，只负责一个元素的获取，从b中取出(k[1], j)存入共享内存b中的(pi, pj)位置
            b_shared[pi, pj] = b_storage[b_pos]

        # 同步线程
        cuda.syncthreads()

        # a_shape[2]（等于b_shape[1]）在a和b为三维情况下，刚好是要相乘并求和的元素对的个数
        # 如果a_shape[2]比较大，而BLOCK_DIM比较小，则在k_offset的前几轮循环中都不会越界，只有在最后一轮时才会越界
        # 即(k_offset + kb) > a_shape[2]
        for kb in range(BLOCK_DIM):
            # 判断是否越界
            if (k_offset + kb) < a_shape[2]:
                # a共享内存的pi行 * b共享内存的pj列，并累加起来，k_offset + 32，进入下一轮循环
                acc += a_shared[pi, kb] * b_shared[kb, pj]

    # 在完成上面的for循环后，acc中已经存储着a的第i行乘以b的第j列求和后的结果，将其存放在out的i行j列位置，此线程任务结束
    # 所以对三维张量来说，out的最后两维为i和j的元素，就由全局索引为i和j的线程来负责计算
    if i < out_shape[1] and j < out_shape[2]:
        out_pos = out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j
        out[out_pos] = acc


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)

"""
    # 计算每行有多少个线程
    index_per_grid_x = cuda.blockDim.x * cuda.gridDim.x
    # 计算每个gridDim.z方向有多少个线程数 = 每行的线程数 * 每列的线程数
    index_per_grid = cuda.blockDim.x * cuda.gridDim.x * cuda.blockDim.y * cuda.gridDim.y
    # index_per_grid * cuda.blockIdx.z确定现在是哪“层”（batch），index_per_grid_x * j = 每行的线程数 *（行数-1），最后+i定位了具体的元素
    # 其实就是用三维索引转换为了1-D的序数
    pos = index_per_grid * cuda.blockIdx.z + index_per_grid_x * j + i

    # 对于没有越界的线程
    if pos < out_size:
        # 对没有越界的线程申请内存
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        # 将1-D序数pos转换为输出张量的索引
        to_index(pos, out_shape, out_index)
        # 利用广播规则，将输出张量的索引转换为a张量的索引
        broadcast_index(out_index, out_shape, a_shape, a_index)
        # 利用广播规则，将输出张量的索引转换为b张量的索引
        broadcast_index(out_index, out_shape, b_shape, b_index)
        # a_shape[-1], b_shape[-2]是相等的，就是因为矩阵相乘而被消除掉的维度
        # 同时也代表需要有多少对元素要被相乘累加
        for j in range(a_shape[-1]):
            temp_j = j + 0
            # 遍历a的行
            a_index[len(a_shape) - 1] = temp_j
            # 将a的索引变为1-D序数
            pos_a = index_to_position(a_index, a_strides)
            # 遍历b的列
            b_index[len(b_shape) - 2] = temp_j
            # 将a的索引变为1-D序数
            pos_b = index_to_position(b_index, b_strides)
            # 将取出的元素两两相乘并累加（out本来就是零张量，0不影响累加）
            out[pos] += (a_storage[pos_a] * b_storage[pos_b])


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
"""


