"""Tests for general operation definitions."""

from typing import Type

import numpy as onp
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import numpy as jnp
from utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform_batch,
)

import jaxlie
from jaxlie.utils import autobatch

shape = (3, 1, 5)  # Try challenging batch shape


@general_group_test
def test_sample_uniform_valid(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that sample_uniform() returns valid group members."""
    T = sample_transform_batch(Group, shape)  # Calls sample_uniform under the hood.
    assert_transforms_close(T, T.normalize())


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so2_from_to_radians_bijective(_random_module):
    """Check that we can convert from and to radians."""
    radians = onp.random.uniform(low=-onp.pi, high=onp.pi)
    assert_arrays_close(jaxlie.SO2.from_radians(radians).as_radians(), radians)


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so3_xyzw_bijective(_random_module):
    """Check that we can convert between xyzw and wxyz quaternions."""
    T = sample_transform_batch(jaxlie.SO3, shape)
    assert_transforms_close(T, jaxlie.SO3.from_quaternion_xyzw(T.as_quaternion_xyzw()))


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so3_rpy_bijective(_random_module):
    """Check that we can convert between quaternions and Euler angles."""
    T = sample_transform_batch(jaxlie.SO3, shape)
    assert_transforms_close(T, jaxlie.SO3.from_rpy_radians(*T.as_rpy_radians()))


@general_group_test
def test_log_exp_bijective(Group: Type[jaxlie.MatrixLieGroup]):
    """Check 1-to-1 mapping for log <=> exp operations."""
    transform = sample_transform_batch(Group, shape)

    tangent = transform.log()
    assert tangent.shape[-1] == Group.tangent_dim

    exp_transform = Group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    assert_arrays_close(tangent, exp_transform.log())


@general_group_test
def test_inverse_bijective(Group: Type[jaxlie.MatrixLieGroup]):
    """Check inverse of inverse."""
    transform = sample_transform_batch(Group, shape)
    assert_transforms_close(transform, transform.inverse().inverse())


@general_group_test
def test_matrix_bijective(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that we can convert to and from matrices."""
    transform = sample_transform_batch(Group, shape)
    assert_transforms_close(transform, Group.from_matrix(transform.as_matrix()))


@general_group_test
def test_adjoint(Group: Type[jaxlie.MatrixLieGroup]):
    """Check adjoint definition."""
    transform = sample_transform_batch(Group, shape)
    omega = onp.random.randn(Group.tangent_dim)
    assert_transforms_close(
        autobatch(lambda t: t @ Group.exp(omega))(transform),
        autobatch(lambda t: Group.exp(t.adjoint() @ omega) @ t)(transform),
    )


@general_group_test
def test_repr(Group: Type[jaxlie.MatrixLieGroup]):
    """Smoke test for __repr__ implementations."""
    transform = sample_transform_batch(Group, shape)
    print(transform)


@general_group_test
def test_apply(Group: Type[jaxlie.MatrixLieGroup]):
    """Check group action interfaces."""
    T_w_b = sample_transform_batch(Group, shape)
    p_b = onp.random.randn(Group.space_dim)

    if Group.matrix_dim == Group.space_dim:
        assert_arrays_close(
            autobatch(lambda T: T @ p_b)(T_w_b),
            autobatch(lambda T: T.apply(p_b))(T_w_b),
            autobatch(lambda T: T.as_matrix() @ p_b)(T_w_b),
        )
    else:
        # Homogeneous coordinates.
        assert Group.matrix_dim == Group.space_dim + 1
        assert_arrays_close(
            autobatch(lambda T: T @ p_b)(T_w_b),
            autobatch(lambda T: T.apply(p_b))(T_w_b),
            autobatch(lambda T: (T.as_matrix() @ onp.append(p_b, 1.0))[:-1])(T_w_b),
        )


@general_group_test
def test_multiply(Group: Type[jaxlie.MatrixLieGroup]):
    """Check multiply interfaces."""
    T_w_b = sample_transform_batch(Group, shape)
    T_b_a = sample_transform_batch(Group, shape)
    assert_arrays_close(
        autobatch(lambda x: x.as_matrix() @ x.inverse().as_matrix())(T_w_b),
        autobatch(lambda x: x.as_matrix() @ jnp.linalg.inv(x.as_matrix()))(T_w_b),
        onp.tile(onp.eye(Group.matrix_dim), (*shape, 1, 1)),
    )
    assert_transforms_close(
        autobatch(lambda x, y: x @ y)(T_w_b, T_b_a),
        autobatch(lambda x, y: Group.multiply(x, y))(T_w_b, T_b_a),
    )
