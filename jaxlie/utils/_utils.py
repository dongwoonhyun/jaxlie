from typing import TYPE_CHECKING, Callable, Type, TypeVar

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax import vmap, lax
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
        if isinstance(arg, jax.Array):
            return len(arg.shape) > data_rank
        elif isinstance(arg, jdc.EnforcedAnnotationsMixin):
            return len(arg.get_batch_axes()) > 0
    raise ValueError("No arrays found in arguments.")


def autobatch(fn, data_rank=1, use_vmap=True):
    """Magical method decorator to automatically batch over any batch dimensions.

    vmap is useful for vectorizing small batch dimensions, but requires a lot of memory.
    lax.map is slower but uses less memory.
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        if is_batched(*args, data_rank=data_rank, **kwargs):
            if use_vmap:
                return vmap(lambda a, k: wrapped(*a, **k))(args, kwargs)
            else:
                return lax.map(lambda x: wrapped(*x[0], **x[1]), (args, kwargs))
        else:
            if use_vmap:
                return fn(*args, **kwargs)
            else:
                return jax.checkpoint(fn)(*args, **kwargs)

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


def stack(objs, axis=0, data_rank=1):
    """Stack a list of objects into a single object with an extra batch dimension."""
    if axis < 0:  # For negative axis, skip over data dimensions.
        axis -= data_rank
    return objs[0].__class__(
        **{
            f.name: jnp.stack([o.__dict__[f.name] for o in objs], axis=axis)
            for f in jdc.fields(objs[0])
        }
    )


def concatenate(objs, axis=0, data_rank=1):
    """Concatenate a list of objects into a single object with an extra batch dimension."""
    if axis < 0:  # For negative axis, skip over data dimensions.
        axis -= data_rank
    return objs[0].__class__(
        **{
            f.name: jnp.concatenate([o.__dict__[f.name] for o in objs], axis=axis)
            for f in jdc.fields(objs[0])
        }
    )
