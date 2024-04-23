from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `weight` tensor.
        weight_strides (Strides): strides for `weight` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    # 因为是weight @ input，所以weight的第一维是out_channels_
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # 并行遍历输出张量的每一个元素，并计算填充
    for out_i in prange(out_size):
        # 申请内存要在循环内，这样才能保证每个线程有独立的空间
        out_index = out_shape.copy()
        # 将1-D序数out_i转换为索引
        to_index(out_i, out_shape, out_index)
        # 提取出当前正在处理的批次、输出通道和宽度信息
        current_batch, current_out_channels, current_pos = out_index
        # 用于累加卷积结果
        acc = 0
        # 遍历输入通道
        # 对于每一个out中的元素，都是需要输入的每个通道和输入通道相对应权重相乘求和得来
        for current_in_channels in prange(in_channels):
            # 遍历卷积核宽度方向
            for current_kw in prange(kw):
                weight_idx = np.array([current_out_channels, current_in_channels, current_kw])
                in_idx = 0
                accumulate = False
                # reverse用于决定卷积核的遍历方向（左->右 or 左<-右）
                # 注意：从左往右遍历时，填充0，填充在输入张量的右边
                # 如果是左->右，且发生越界则不累加，相当于在右边补0
                # 如果是左->右，且没有越界则：
                if (not reverse) and (current_pos + current_kw < width):
                    in_idx = index_to_position(
                        np.array([current_batch, current_in_channels, current_pos + current_kw]), s1
                    )
                    accumulate = True
                # 注意：从右往左遍历时，填充0，填充在输入张量的左边
                # 如果是左<-右，且发生越界则不累加，相当于在左边补0
                # 如果是左<-右，且没有越界则：
                elif reverse and (current_pos - current_kw >= 0):
                    in_idx = index_to_position(
                        np.array([current_batch, current_in_channels, current_pos - current_kw]), s1
                    )
                    accumulate = True
                # 当填充0的情况发生时，不用累加
                if accumulate:
                    acc += input[in_idx] * weight[index_to_position(weight_idx, s2)]
        # 在完成遍历后，将累加结果存到out相应位置
        out[index_to_position(out_index, out_strides)] = acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # 目标：遍历并填充out中的每个元素
    for out_i in prange(out_size):
        out_index = out_shape.copy()
        to_index(out_i, out_shape, out_index)
        current_batch, current_out_channels, current_height, current_pos = out_index
        acc = 0.0

        # 遍历计算
        for current_in_channels in range(in_channels):
            for current_kh in range(kh):
                for current_kw in range(kw):
                    # reverse为False时，hw两个维度都从0开始遍历，否则就从另一端开始
                    current_h, current_w = (
                        (kh - current_kh - 1, kw - current_kw - 1) if reverse else (current_kh, current_kw)
                    )

                    # 将权重的索引转换为1-D序数
                    weight_idx = (
                            (s20 * current_out_channels) + (s21 * current_in_channels) + (s22 * current_h) + (s23 * current_w)
                    )
                    # 通过序数将权重取出来
                    curr_weight = weight[weight_idx]

                    in_idx = 0
                    accumulate = False

                    # 基本和conv1d一样
                    if reverse and (current_height - current_h >= 0) and (current_pos - current_w >= 0):
                        in_idx = (
                                (current_batch * s10)
                                + (current_in_channels * s11)
                                + (current_height - current_h) * s12
                                + (current_pos - current_w) * s13
                        )
                        accumulate = True

                    elif (
                            (not reverse)
                            and (current_h + current_height < height)
                            and (current_w + current_pos < width)
                    ):
                        in_idx = (
                                (current_batch * s10)
                                + (current_in_channels * s11)
                                + (current_height + current_h) * s12
                                + (current_pos + current_w) * s13
                        )
                        accumulate = True

                    if accumulate:
                        acc += curr_weight * input[in_idx]

        out_loc = (
                (current_batch * out_strides[0])
                + (current_out_channels * out_strides[1])
                + (current_height * out_strides[2])
                + (current_pos * out_strides[3])
        )
        out[out_loc] = acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
