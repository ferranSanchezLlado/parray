import functools
import math
from functools import reduce
from typing import TypeAlias, TypeVar, Callable, Generic, Any, Iterator, cast

T = TypeVar("T")
_recursiveList: TypeAlias = list["_recursiveList"] | T
RecursiveList: TypeAlias = list[_recursiveList]


def array_equal(x: Any, y: Any) -> bool:
    if x is y:
        return True
    if not isinstance(x, array) and not isinstance(y, array):
        return cast(bool, x == y)
    if x.shape != y.shape:
        return False
    return all(array_equal(a, b) for a, b in zip(x, y))


def flatten(data: "RecursiveList | array[T]") -> list[T]:
    # if isinstance(data, array):
    #     data = data.data
    # if isinstance(data, list):
    #     return functools.reduce(lambda x, y: x + flatten(y), data, [])
    # return [data]
    if isinstance(data, (list, array)):
        return [item for sublist in data for item in flatten(sublist)]
    return [data]


def _is_invalid_shape(data: "array[T]", shape: tuple[int, ...]) -> bool:
    if len(shape) == 0:
        return not isinstance(data, array)
    if len(shape) == 1:
        return shape[0] != len(data)
    return any(_is_invalid_shape(row, shape[1:]) for row in data)


def _unravel_index(index: int, shape: tuple[int, ...]) -> list[Any]:  # Recursive list of ints like [int, ...]
    if len(shape) == 1:
        return [index]
    return [index // reduce(lambda x, y: x * y, shape[1:], 1)] + _unravel_index(
        index % reduce(lambda x, y: x * y, shape[1:], 1), shape[1:]
    )


def unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(_unravel_index(index, shape))


def _max_op(x: "array[T] | T", y: "array[T] | T") -> "array[T] | T":
    if isinstance(x, array) and isinstance(y, array):
        return array([max(a, b) for a, b in zip(x, y)])
    return max(x, y)


def log(x: "array[T] | T") -> "array[T] | T":
    if isinstance(x, array):
        return array([log(el) for el in x])
    return math.log(x)


class array(Generic[T]):
    def __init__(self, data: "array[T] | RecursiveList", shape: tuple[int, ...] | None = None):
        if shape is not None:
            data = flatten(data)
            size = len(data)
            if len(shape) != 1:
                size_element = reduce(lambda x, y: x * y, shape[1:], 1)
                data = [array(data[i * size_element: (i + 1) * size_element], shape[1:]) for i in range(shape[0])]
        else:
            data = [array(x, None) if isinstance(x, list) else x for x in data]
            shape = tuple([len(data)] + list(data[0].shape if isinstance(data[0], array) else []))
            size = reduce(lambda x, y: x * y, shape)

        if _is_invalid_shape(data, shape):  # type: ignore
            raise ValueError(f"The shape {shape} is invalid for the data {data}")
        self.data: list[array[T] | T] = data
        self.shape = shape
        self._size = size

    def _get(self, key: int | tuple[int, ...] | slice) -> "array[T] | T | list[array[T] | T]":
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if isinstance(key, (int, slice)):
            return self.data[key]
        if isinstance(key[0], slice):
            slice_indices = key[0].indices(self.shape[0])
            return [self.data[i]._get(key[1:]) for i in range(*slice_indices)]
        return self.data[key[0]]._get(key[1:])

    def __getitem__(self, key: int | tuple[int, ...] | slice) -> "array[T] | T | list[array[T] | T]":
        data = self._get(key)
        return data if not isinstance(data, list) else array(data)

    def __setitem__(self, key: int | tuple[int, ...] | slice, value: T) -> None:
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if isinstance(key, int):
            self.data[key] = value
        elif isinstance(key, slice):
            for index, item in zip(range(*key.indices(self.shape[0])), value):
                self.data[index] = item
        else:
            if isinstance(key[0], slice):
                for index, item in zip(range(*key[0].indices(self.shape[0])), value):
                    self.data[index][key[1]] = item
            self[key[0]][key[1:]] = value

    def __str__(self) -> str:
        return str(self.tolist())

    def __repr__(self) -> str:
        return f"array({self.data}, {self.shape})"

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator["array[T] | T"]:
        return self.data.__iter__()

    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> "array[T]":
        if self.ndim() == 1:
            return self
        elif self.ndim() == 2:
            return array([[self[i, j] for i in range(self.shape[0])] for j in range(self.shape[1])],
                         shape=(self.shape[1], self.shape[0]))
        raise NotImplementedError("T is not implemented for ndim > 2")

    def __fix_dimensions_basic_op(self, other: Any) -> "tuple[array[T], array[T]]":
        if self.ndim() == 1:
            shape = (1, self.shape[0]) if other.shape[0] == 1 else (self.shape[0], 1)
            self = self.reshape(*shape)
        if other.ndim() == 1:
            shape = (other.shape[0], 1) if self.shape[1] == 1 else (1, other.shape[0])
            other = other.reshape(*shape)
        return self, other

    def __basic_op(self, other: Any, op: "Callable[[T, T], T]") -> "array[T]":
        if not isinstance(other, array):
            return array([op(el, other) for el in self])
        if self.shape == other.shape:
            return array([op(a, b) for a, b in zip(self, other)])
        if self.ndim() == 1 or other.ndim() == 1:  # TODO: Check if this case is necessary
            flat, multi = self, other
            if other.ndim() == 1:
                flat, multi = other, self
            if all(x == 1 for x in multi.shape[1:]):
                multi = multi.T
            if flat.shape[0] == multi.shape[-1] and all(x == 1 for x in multi.shape[:-1]):
                return array([op(a, b) for a, b in zip(flat, flatten(multi))], multi.shape)
            raise ValueError(f"Shapes {self.shape} and {other.shape} are not broadcastable")
        self, other = self.__fix_dimensions_basic_op(other)
        if self.ndim() == 2 and other.ndim() == 2 and (
                self.shape[0] == other.shape[0] and (self.shape[1] == 1 or other.shape[1] == 1) or self.shape[1] ==
                other.shape[1] and (self.shape[0] == 1 or other.shape[0] == 1)):
            if self.shape[0] == other.shape[0] and self.shape[1] == 1:
                return array(
                    [[op(self[i, 0], other[i, j]) for j in range(other.shape[1])] for i in range(self.shape[0])],)
            if self.shape[0] == other.shape[0] and other.shape[1] == 1:
                return array(
                    [[op(self[i, j], other[i, 0]) for j in range(self.shape[1])] for i in range(self.shape[0])])
            if self.shape[1] == other.shape[1] and self.shape[0] == 1:
                return array(
                    [[op(self[0, j], other[i, j]) for j in range(self.shape[1])] for i in range(other.shape[0])])
            if self.shape[1] == other.shape[1] and other.shape[0] == 1:
                return array(
                    [[op(self[i, j], other[0, j]) for j in range(self.shape[1])] for i in range(self.shape[0])])
        raise ValueError(f"Operands could not be broadcast together with shapes: {self.shape} {other.shape}")

    def __add__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x + y)

    def __radd__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y + x)

    def __sub__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x - y)

    def __rsub__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y - x)

    def __mul__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x * y)

    def __rmul__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y * x)

    def __truediv__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x / y)

    def __rtruediv__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y / x)

    def __floordiv__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x // y)

    def __rfloordiv__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y // x)

    def __mod__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x % y)

    def __rmod__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y % x)

    def __pow__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: x ** y)

    def __rpow__(self, other: Any) -> "array[T]":
        return self.__basic_op(other, lambda x, y: y ** x)

    def __neg__(self) -> "array[T]":
        return -1 * self

    def __abs__(self) -> "array[T]":
        return array([abs(x) for x in self.data], self.shape)

    def __fix_1dim(self, other: Any) -> "array[T]":
        if self.ndim() == 1:
            shape = (1, self.shape[0]) if other.shape[0] != 1 else (self.shape[0], 1)
            self = self.reshape(*shape)
        if other.ndim() == 1:
            shape = (other.shape[0], 1) if self.shape[1] != 1 else (1, other.shape[0])
            other = other.reshape(*shape)
        return self, other

    def __matmul__(self, other: Any) -> "array[T]":
        if not isinstance(other, array):
            raise TypeError("can only matmul with array")
        if self.ndim() == 1 and other.ndim() == 1:
            return array((self * other).sum(), shape=(1,))
        self, other = self.__fix_1dim(other)
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"shape mismatch: {self.shape} {other.shape}")
        return array(
            [[sum((self[i, k] * other[k, j] for k in range(self.shape[1]))) for j in range(other.shape[1])] for i in
             range(self.shape[0])])

    def tolist(self) -> RecursiveList:
        if self.ndim() == 1:
            return self.data
        return [a.tolist() for a in self.data]

    def reshape(self, *shape: int) -> "array[T]":
        return array(self, shape)

    def flatten(self) -> "array[T]":
        return array(flatten(self))

    def __accumulator_op(self, axis: int | None,
                         op: Callable[["array[T] | T", "array[T] | T"], "array[T] | T"]) -> "array[T] | T":
        if axis is None:
            return functools.reduce(op, flatten(self) if self.ndim() > 1 else self.data)
        if axis > self.ndim() - 1:
            raise ValueError("axis out of range")
        if axis + 1 == self.ndim() == 1:  # 1d array
            return functools.reduce(op, self.data)
        slice_: list[slice | int] = [slice(None)] * axis
        axis_data = [self[tuple(slice_ + [i])] for i in range(self.shape[axis])]
        return array(functools.reduce(op, axis_data))

    def sum(self, axis: int | None = None) -> "array[T] | T":
        return self.__accumulator_op(axis, lambda x, y: x + y)

    def max(self, axis: int | None = None) -> "array[T] | T":
        return self.__accumulator_op(axis, _max_op)

    def argmax(self, axis: int | None = None) -> tuple[int, ...]:
        if axis is None:
            return unravel_index(flatten(self).index(self.max()), self.shape)
        if axis > self.ndim() - 1:
            raise ValueError("axis out of range")
        if axis + 1 == self.ndim() == 1:  # 1d array
            return self.data.index(self.max()),
        if self.ndim() > 2:
            raise NotImplementedError
        other = self.T if axis == 0 else self
        return tuple([other[i].argmax()[0] for i in range(other.shape[0])])

    def append(self, other: "array[T]") -> "array[T]":
        if self.shape[1:] == other.shape[1:]:
            self.data.extend(other.data)
            self.shape = (self.shape[0] + other.shape[0], *self.shape[1:])
            return self
        if self.shape[1:] != other.shape:
            raise ValueError(f"shape mismatch: {self.shape} {other.shape}")
        self.data.append(other)
        self.shape = (self.shape[0] + 1, *self.shape[1:])
        return self

    # TODO: Testing
    def __eq__(self, other: Any) -> "array[bool]":
        return self.__basic_op(other, lambda x, y: x == y)

    def __lt__(self, other: Any) -> "array[bool]":
        return self.__basic_op(other, lambda x, y: x < y)

    def __gt__(self, other: Any) -> "array[bool]":
        return self.__basic_op(other, lambda x, y: x > y)

    def all(self) -> bool:
        return all(flatten(self))

    def any(self) -> bool:
        return any(flatten(self))
