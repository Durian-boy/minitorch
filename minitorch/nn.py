from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    # 确保输入图像的高度和宽度可以被池化核的高度和宽度整除
    # 这是进行池化操作的前提条件，否则无法将图像整齐地划分为不重叠的池化区域
    assert height % kh == 0
    assert width % kw == 0

    # 计算新的高度和宽度
    new_h, new_w = (height // kh, width // kw)
    inp = (
        input.contiguous()
        .view(batch, channel, new_h, kh, new_w, kw)  # 先将height和width展开为(new_h, kh)(new_w, kw)
        .permute(0, 1, 2, 4, 3, 5)  # 再用permute调换顺序
        .contiguous()
        .view(batch, channel, new_h, new_w, kh * kw)
    )
    return inp, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    inp, new_h, new_w = tile(input, kernel)
    # 在dim=4的维度上求平均
    return inp.mean(4).view(batch, channel, new_h, new_w)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    # 这里返回的是一个one-hot张量，如果out中的值与input中的相应值相等，则对应位置为1，否则为0
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        inp, dim = ctx.saved_values
        out = grad_output * argmax(inp, int(dim.item()))
        return (out, 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    exp = input.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    inp, new_height, new_width = tile(input, kernel)
    return max(inp, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # 使用rand(input.shape, backend=input.backend)生成一个与输入张量形状相同的随机张量，其中的值在[0, 1)范围内
    # 将这个随机张量与rate进行比较，得到一个布尔张（每个元素都是True或False）
    # 将这个布尔张量转换为浮点张量（True变为1，False变为0）
    # 最后，将这个浮点张量与原始输入input相乘，从而实现dropout效果
    return (
        input if ignore else (rand(input.shape, backend=input.backend) > rate) * input
    )
