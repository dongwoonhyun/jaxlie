import numpy as onp

from jaxlie import SE3

from jaxlie.utils import reshape, autobatch, stack, concatenate
from jaxlie.utils import stack, concatenate


print("Batch indexing.")

# Create SE3 transforms with batch shape (9, 2)
a_SE3 = SE3(onp.random.random((9, 2, 7)))
assert a_SE3.as_matrix().shape == (9, 2, 4, 4)
assert a_SE3.rotation().as_matrix().shape == (9, 2, 3, 3)

# Index into the batch dimension
assert a_SE3[0].get_batch_axes() == (2,)
assert a_SE3[0, -1].get_batch_axes() == ()
assert a_SE3[:5].get_batch_axes() == (5, 2)
assert a_SE3[:5, 1:].get_batch_axes() == (5, 1)
assert a_SE3[..., 0].get_batch_axes() == (9,)
assert a_SE3[0, ...].get_batch_axes() == (2,)


print("Batch multiplication.")

# Multiply SE3 with batch shape (9, 2) together with an unbatched SE3
b_SE3 = SE3(onp.random.random((7,)))
assert (a_SE3 @ b_SE3).get_batch_axes() == (9, 2)

# Multiply SE3 with batch shape (9, 2) together with SE3 with batch shape (3,)
c_SE3 = SE3(onp.random.random((3, 7)))
assert (a_SE3 @ c_SE3).get_batch_axes() == (9, 2, 3)  # Outer product of batch
assert (c_SE3 @ a_SE3).get_batch_axes() == (3, 9, 2)  # Outer product of batch

# Elementwise multiplication of two SE3s with identical shape using explicit autobatch
assert autobatch(lambda a, b: a @ b)(a_SE3, a_SE3).get_batch_axes() == (9, 2)

# Apply SE3 with batch shape (9, 2) to a single point
p = onp.random.randn(3)
assert (a_SE3 @ p).shape == (9, 2, 3)

# Apply SE3 with batch shape (9, 2) to a point with batch shape (200, 100)
p = onp.random.randn(200, 100, 3)
assert (a_SE3 @ p).shape == (9, 2, 200, 100, 3)


print("Batch shape manipulation.")

# Reshape batch shape
assert reshape(a_SE3, (18,)).get_batch_axes() == (18,)
assert reshape(a_SE3, (3, 3, 2)).get_batch_axes() == (3, 3, 2)
# Stack 10 SE3s with the same batch shape (3, 5, 2)
list_SE3 = [SE3(onp.random.random((3, 5, 2, 7))) for _ in range(10)]
assert stack(list_SE3, 0).get_batch_axes() == (10, 3, 5, 2)
assert stack(list_SE3, 1).get_batch_axes() == (3, 10, 5, 2)
assert stack(list_SE3, -1).get_batch_axes() == (3, 5, 2, 10)
# Concatenate
assert concatenate(list_SE3, 0).get_batch_axes() == (30, 5, 2)
assert concatenate(list_SE3, 1).get_batch_axes() == (3, 50, 2)
assert concatenate(list_SE3, -1).get_batch_axes() == (3, 5, 20)

print("Other SE3 examples with batching.")

# We can compute a w<-b transform by integrating over an se(3) screw, equivalent
# to `SE3.from_matrix(expm(wedge(twist)))`.
twist = onp.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0])
twist = onp.tile(twist, (2, 1, 3, 1))  # Add batch dimensions (2, 1, 3)
T_w_b = SE3.exp(twist)

# Compute inverses:
T_b_w = T_w_b.inverse()

# Batch-element-wise multiplication
identity = autobatch(lambda a, b: a @ b)(T_w_b, T_b_w)

# Compute adjoints:
adjoint_T_w_b = T_w_b.adjoint()
print(f"\t{adjoint_T_w_b=}")

# Recover our twist, equivalent to `vee(logm(T_w_b.as_matrix()))`:
recovered_twist = T_w_b.log()
print(f"\t{recovered_twist=}")
