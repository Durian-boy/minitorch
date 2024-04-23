from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    sum = 0
    for index, strides in zip(index, strides):
        sum += index * strides
    return sum



def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    cur_orf: int = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        cur_shape: int = shape[i]
        out_index[i] = int(cur_orf % cur_shape)
        cur_orf = cur_orf // cur_shape

    """ 使用如下代码会对np.prod函数报错
    # 这里加0也是为了避免numba报错
    ordinal = ordinal + 0
    for i, s in enumerate(shape):
        # 这里如果用operator.py中实现的prod，numba会报错：没有定义prod
        product = np.prod(shape[i:])
        divisor = product / s  # 因为在计算strides的时候多乘了s，要去掉s
        index = int(ordinal // divisor)

        ordinal -= index * divisor
        out_index[i] = index"""


# 该函数的作用：在实际张量运算中我们并不需要真的把张量复制扩展为big_shape，我们只需要将
# big_shape的索引转换为shape的索引来找到对应的元素参加计算，这个函数就是完成找对应元素的任务
def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # 最终输出的out_index和shape规模一致，这是因为big_shape比shape多出来的维度都可以通过
    # 在shape左边加维度并广播（复制）来实现，既然是复制shape来实现，那么这些维度其实并不重要
    # 我们只需要关注shape即可
    for i in range(len(shape)):
        offset = i + len(big_shape) - len(shape)
        # 如果shape中某一维度为1，无论这个维度是否有广播，都可以取0来访问想要的元素
        # 如果shape中某个维度不为1，则说明这个维度没有被广播，只能用原索引来访问
        out_index[i] = big_index[offset] if shape[i] != 1 else 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # 先补齐维度，再考虑是否具备广播的条件
    length = max(len(shape1), len(shape2))
    if len(shape1) > len(shape2):
        shape2 = [1 for i in range(length - len(shape2))] + list(shape2)
    else:
        shape1 = [1 for i in range(length - len(shape1))] + list(shape1)

    # 开始验证是否具备广播的条件
    ans = []
    for i in range(length):
        # 在二者某维度取值不同时，只有某一方取值为1，才符合广播的条件
        if shape1[i] != shape2[i] and shape1[i] != 1 and shape2[i] != 1:
            raise IndexingError('violation of broadcasting rules')
        # 若符合广播条件，将二者中较大的值存入ans，作为union matrix某维度的取值
        ans.append(max(shape1[i], shape2[i]))
    return tuple(ans)



def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    # 将张量数据存储在CUDA设备上
    def to_cuda_(self) -> None:  # pragma: no cover
        # 检查张量的存储是否已经在CUDA设备上
        if not numba.cuda.is_cuda_array(self._storage):
            # 如果不在则：将数据移到CUDA设备上
            self._storage = numba.cuda.to_device(self._storage)

    # 用于判断该张量在内存中是否连续
    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # permute可以这么理解：每次只交换两个索引的值（transpose），保证其他索引值不变，直到完成所有索引的变换
        return TensorData(
            self._storage,
            tuple([self.shape[i] for i in order]),
            tuple([self._strides[i] for i in order]),
        )


    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
