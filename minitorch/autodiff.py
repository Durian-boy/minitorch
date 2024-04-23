from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    my_val = list(vals)
    my_val[arg] += epsilon
    f1 = f(*my_val)
    my_val[arg] -= 2 * epsilon
    f2 = f(*my_val)
    return (f1 - f2) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # # 有向无环图的拓扑排序->寻找入度为0的节点->倒序
    Visited = []
    result = []
    def visit(n: Variable):
        if n.is_constant():  # 常量不做处理
            return
        if n.unique_id in Visited:  # 已经访问过了
            return
        if not n.is_leaf():
            for input in n.history.inputs:
                visit(input)
        Visited.append(n.unique_id)  # 该节点已经被访问
        result.insert(0, n)

    visit(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    result = topological_sort(variable)
    # 得到以Variable为最右节点的子节点拓扑图
    node2deriv = {}
    node2deriv[variable.unique_id] = deriv
    for n in result:
        if n.is_leaf():
            continue
        if n.unique_id in node2deriv.keys():
            deriv = node2deriv[n.unique_id]
        deriv_tmp = n.chain_rule(deriv)
        for key, item in deriv_tmp:
            # 叶结点的导数被累加至derivative属性中
            if key.is_leaf():
                key.accumulate_derivative(item)
                continue
            # 中间变量的导数被存储在node2deriv中
            if key.unique_id in node2deriv.keys():
                # 根据 unique_id 判断是否已经计算得到部分梯度
                # !不要使用Scallar in，因为重构了__eq__
                # 中间变量也有复用的情况，所以需要累加
                node2deriv[key.unique_id] += item
            else:
                node2deriv[key.unique_id] = item


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
