from typing import TYPE_CHECKING, Callable, Type, TypeVar

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax import vmap
from functools import wraps

if TYPE_CHECKING:
    from .._base import MatrixLieGroup


T = TypeVar("T", bound="MatrixLieGroup")


def get_epsilon(dtype: jnp.dtype) -> float:
    """Helper for grabbing type-specific precision constants.

    Args:
        dtype: Datatype.

    Returns:
        Output float.
    """
    return {
        jnp.dtype("float32"): 1e-5,
        jnp.dtype("float64"): 1e-10,
    }[dtype]


def register_lie_group(
    *,
    matrix_dim: int,
    parameters_dim: int,
    tangent_dim: int,
    space_dim: int,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Lie group dataclasses.

    Sets dimensionality class variables, and marks all methods for JIT compilation.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes.
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

        # JIT all methods.
        for f in filter(
            lambda f: not f.startswith("_")
            and callable(getattr(cls, f))
            and f != "get_batch_axes",  # Avoid returning tracers.
            dir(cls),
        ):
            setattr(cls, f, jax.jit(getattr(cls, f)))

        return cls

    return _wrap


def is_batched(*args, data_rank=1, **kwargs):
    """Returns True if any of the arguments are batched.

    data_rank is the rank of the underlying data (i.e. non-batch dimensions)."""
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, jnp.ndarray):
            return len(arg.shape) > data_rank
        elif isinstance(arg, jdc.EnforcedAnnotationsMixin):
            return len(arg.get_batch_axes()) > 0
    raise ValueError("No arrays found in arguments.")


def autobatch(fn, data_rank=1):
    """Magical method decorator to automatically vmap over any batch dimensions."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        if is_batched(*args, data_rank=data_rank, **kwargs):
            return vmap(lambda a, k: wrapped(*a, **k))(args, kwargs)
        else:
            return fn(*args, **kwargs)

    return wrapped


def reshape(obj, batch_shape, data_rank=1):
    """Create a new instance of the same class, with reshaped batch axes."""
    return obj.__class__(
        **{
            f.name: obj.__dict__[f.name].reshape(
                [*batch_shape, *obj.__dict__[f.name].shape[-data_rank:]]
            )
            for f in jdc.fields(obj)
        }
    )
