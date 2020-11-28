import dataclasses
from typing import Type

from jax import numpy as jnp
from overrides import overrides

from ._base import MatrixLieGroup
from ._types import Matrix, TangentVector, Vector
from ._utils import register_lie_group


@register_lie_group
@dataclasses.dataclass(frozen=True)
class SO2(MatrixLieGroup):

    # SO2-specific
    unit_complex: Vector

    @staticmethod
    def from_theta(theta: float) -> "SO2":
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    # Factory

    @classmethod
    @overrides
    def identity(cls: Type["SO2"]) -> "SO2":
        return SO2(unit_complex=jnp.array([1.0, 0.0]))

    # Accessors

    @staticmethod
    @overrides
    def matrix_dim() -> int:
        return 2

    @staticmethod
    @overrides
    def compact_dim() -> int:
        return 2

    @staticmethod
    @overrides
    def tangent_dim() -> int:
        return 1

    @overrides
    def matrix(self) -> Matrix:
        cos, sin = self.unit_complex
        return jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

    @overrides
    def compact(self) -> Vector:
        return self.unit_complex

    # Operations

    @overrides
    def apply(self: "SO2", target: Vector) -> Vector:
        assert target.shape == (2,)
        return self.matrix() @ target

    @overrides
    def product(self: "SO2", other: "SO2") -> "SO2":
        return SO2(self.matrix() @ self.unit_complex)

    @staticmethod
    @overrides
    def exp(tangent: TangentVector) -> "SO2":
        assert tangent.shape == (1,)
        cos = jnp.cos(tangent[0])
        sin = jnp.sin(tangent[0])
        return SO2(unit_complex=jnp.array([cos, sin]))

    @staticmethod
    @overrides
    def log(self: "SO2") -> TangentVector:
        return jnp.arctan2(self.unit_complex[1, None], self.unit_complex[0, None])

    @staticmethod
    @overrides
    def inverse(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex.at[1].set(-self.unit_complex[1]))

    @overrides
    def normalize(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex / jnp.linalg.norm(self.unit_complex))