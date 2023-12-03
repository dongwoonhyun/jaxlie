import abc
from typing import ClassVar, Generic, Tuple, Type, TypeVar, Union, overload

import jax
import numpy as onp
from typing_extensions import final, override
import jax_dataclasses as jdc

from . import hints
from .utils import autobatch, tile

GroupType = TypeVar("GroupType", bound="MatrixLieGroup")
SEGroupType = TypeVar("SEGroupType", bound="SEBase")


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties.
    # > These will be set in `_utils.register_lie_group()`.

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(
        # Notes:
        # - For the constructor signature to be consistent with subclasses, `parameters`
        #   should be marked as positional-only. But this isn't possible in Python 3.7.
        # - This method is implicitly overriden by the dataclass decorator and
        #   should _not_ be marked abstract.
        self,
        parameters: jax.Array,
    ):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    # Shared implementations.

    def __getitem__(self, key):
        """Allow retrieving a subset of the aperture using [] indexing operator."""
        if hasattr(key, "__iter__") and any([k == Ellipsis for k in key]):
            # If the key has an ellipsis (i.e. "..."), make sure the data dimension is
            # explicitly included via slice(None) (i.e. ":"). This assumes the actual
            # data rank is always 1.
            key = (*key, slice(None))
        return self.__class__(
            **{f.name: self.__dict__[f.name][key] for f in jdc.fields(self)}
        )

    def __len__(self):
        """Return the number of elements."""
        return 1 if self.get_batch_axes() == () else self.wxyz_xyz.shape[0]

    def __iter__(self):
        """Default iterator over elements."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        """Pretty-printing."""
        data = {
            n: jax.numpy.reshape(self.__dict__[n], (-1, self.__dict__[n].shape[-1]))
            for f in jdc.fields(self)
            if (n := f.name)
        }
        str = f"{self.__class__.__name__}(batch_axes={self.get_batch_axes()},"
        # If size is too large, only print the first and last few elements
        n = data[list(data)[0]].shape[0]
        ranges = [range(n)] if n <= 10 else [range(5), None, range(n - 4, n)]
        for r in ranges:
            if r is None:
                str += "\n..."
            else:
                # Print each element in the batch
                for i in r:
                    str += "\n%5d: " % i
                    for n, d in data.items():
                        str += "%s=[" % n
                        for e in d[i]:
                            try:  # When a concrete value, print value
                                str += " %+7.4f, " % e
                            except:  # When an abstract tracer, print dtype
                                str += " %7s, " % e.dtype
                        str = str[:-2] + "], "
        str = str[:-2] + ")"
        return str

    def _reshape(self, batch_shape):
        """Create a new instance of the same class, with reshaped batch axes.

        The leading underscore is apparently necessary to avoid jax tracing.
        """
        return self.__class__(
            **{
                n: self.__dict__[n].reshape([*batch_shape, self.__dict__[n].shape[-1]])
                for f in jdc.fields(self)  # Loop over jax_dataclass fields
                if (n := f.name)  # Assign temporary variable
            }
        )

    @overload
    def __matmul__(self: GroupType, other: GroupType) -> GroupType:
        ...

    @overload
    def __matmul__(self, other: hints.Array) -> jax.Array:
        ...

    def __matmul__(
        self: GroupType, other: Union[GroupType, hints.Array]
    ) -> Union[GroupType, jax.Array]:
        """Overload for the `@` operator.

        Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`.
        """
        if isinstance(other, (onp.ndarray, jax.Array)):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)
        else:
            assert False, f"Invalid argument type for `@` operator: {type(other)}"

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(cls: Type[GroupType]) -> GroupType:
        """Returns identity element.

        Returns:
            Identity element.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls: Type[GroupType], matrix: hints.Array) -> GroupType:
        """Get group member from matrix representation.

        Args:
            matrix: Matrix representaiton.

        Returns:
            Group member.
        """

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> jax.Array:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> jax.Array:
        """Get underlying representation."""

    # Operations.

    @abc.abstractmethod
    def apply(self, target: hints.Array) -> jax.Array:
        """Applies group action to a point.

        Args:
            target: Point to transform.

        Returns:
            Transformed point.
        """

    @abc.abstractmethod
    def multiply(self: GroupType, other: GroupType) -> GroupType:
        """Composes this transformation with another.

        Returns:
            self @ other
        """

    @classmethod
    @abc.abstractmethod
    def exp(cls: Type[GroupType], tangent: hints.Array) -> GroupType:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent: Tangent vector to take the exponential of.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def log(self) -> jax.Array:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self) -> jax.Array:
        """Computes the adjoint, which transforms tangent vectors between tangent
        spaces.

        More precisely, for a transform `GroupType`:
        ```
        GroupType @ exp(omega) = exp(Adj_T @ omega) @ GroupType
        ```

        In robotics, typically used for transforming twists, wrenches, and Jacobians
        across different reference frames.

        Returns:
            Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inverse(self: GroupType) -> GroupType:
        """Computes the inverse of our transform.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def normalize(self: GroupType) -> GroupType:
        """Normalize/projects values and returns.

        Returns:
            GroupType: Normalized group member.
        """

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls: Type[GroupType], key: hints.KeyArray) -> GroupType:
        """Draw a uniform sample from the group. Translations (if applicable) are in the
        range [-1, 1].

        Args:
            key: PRNG key, as returned by `jax.random.PRNGKey()`.

        Returns:
            Sampled group member.
        """

    @abc.abstractmethod
    def get_batch_axes(self) -> Tuple[int, ...]:
        """Return any leading batch axes in contained parameters. If an array of shape
        `(100, 4)` is placed in the wxyz field of an SO3 object, for example, this will
        return `(100,)`.

        This should generally be implemented by `jdc.EnforcedAnnotationsMixin`."""


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)


class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups.

    Each SE(N) group member contains an SO(N) rotation, as well as an N-dimensional
    translation vector.
    """

    # SE-specific interface.

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(
        cls: Type[SEGroupType],
        rotation: ContainedSOType,
        translation: hints.Array,
    ) -> SEGroupType:
        """Construct a rigid transform from a rotation and a translation.

        Args:
            rotation: Rotation term.
            translation: translation term.

        Returns:
            Constructed transformation.
        """

    @classmethod
    def from_rotation(cls: Type[SEGroupType], rotation: ContainedSOType) -> SEGroupType:
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=onp.zeros(cls.space_dim, dtype=rotation.parameters().dtype),
        )

    @classmethod
    def from_translation(
        cls: Type[SEGroupType], translation: hints.Array
    ) -> SEGroupType:
        batch_axes = translation.shape[:-1]
        return cls.from_rotation_and_translation(
            rotation=tile(cls.identity().rotation().__class__.identity(), batch_axes),
            translation=translation,
        )

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> jax.Array:
        """Returns a transform's translation term."""

    # Overrides.

    @final
    @override
    def apply(self, target: hints.Array) -> jax.Array:
        fn = lambda s: s.rotation() @ target + s.translation()
        return autobatch(fn)(self)  # type: ignore

    @final
    @override
    def multiply(self: SEGroupType, other: SEGroupType) -> SEGroupType:
        fn = lambda s: type(s).from_rotation_and_translation(
            rotation=s.rotation() @ other.rotation(),
            translation=(s.rotation() @ other.translation()) + s.translation(),
        )
        return (autobatch)(fn)(self)

    @final
    @autobatch
    @override
    def inverse(self: SEGroupType) -> SEGroupType:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    @final
    @autobatch
    @override
    def normalize(self: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )
