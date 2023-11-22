# jaxlie

This is a fork of the excellent `jaxlie` library for rigid transformations, with added utilities for dealing with the batch axes of MatrixLieGroup objects. The most significant addition is the `autobatch` decorator, which wraps any function to automatically and recursively apply `vmap` across all batch dimensions. Some utilities are also added to facilitate batch axes manipulation. These changes are especially useful for array processing applications (e.g., ultrasound imaging), where you have a large number of sensors with unique poses.

Example usage of new features:

```python

  # Create SE3 transforms with batch shape (9, 2)
  a_SE3 = SE3(onp.random.random((9, 2, 7)))
  assert a_SE3.as_matrix().shape == (9, 2, 4, 4)
  assert a_SE3.rotation().as_matrix().shape == (9, 2, 3, 3)
  # Easily index/slice into an SE3 object's batch dimensions using []
  assert a_SE3[0].get_batch_axes() == (2,)
  assert a_SE3[0, -1].get_batch_axes() == ()
  assert a_SE3[:5].get_batch_axes() == (5, 2)
  assert a_SE3[:5, 1:].get_batch_axes() == (5, 1)
  assert a_SE3[..., 0].get_batch_axes() == (9,)
  assert a_SE3[0, ...].get_batch_axes() == (2,)
  # Multiply SE3 with batch shape (9, 2) together with SE3 with batch shape (3,)
  c_SE3 = SE3(onp.random.random((3, 7)))
  assert (a_SE3 @ c_SE3).get_batch_axes() == (9, 2, 3)  # Outer product of batch
  assert (c_SE3 @ a_SE3).get_batch_axes() == (3, 9, 2)  # Outer product of batch

```

## From the original jaxlie repo

![build](https://github.com/brentyi/jaxlie/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jaxlie/workflows/mypy/badge.svg?branch=master)
![lint](https://github.com/brentyi/jaxlie/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jaxlie/branch/master/graph/badge.svg)](https://codecov.io/gh/brentyi/jaxlie)
[![pypi_dowlnoads](https://pepy.tech/badge/jaxlie)](https://pypi.org/project/jaxlie)

**[ [API reference](https://brentyi.github.io/jaxlie) ]** **[
[PyPI](https://pypi.org/project/jaxlie/) ]**

`jaxlie` is a library containing implementations of Lie groups commonly used for
rigid body transformations, targeted at computer vision &amp; robotics
applications written in JAX. Heavily inspired by the C++ library
[Sophus](https://github.com/strasdat/Sophus).

We implement Lie groups as high-level (data)classes:

<table>
  <thead>
    <tr>
      <th>Group</th>
      <th>Description</th>
      <th>Parameterization</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td><code>jaxlie.<strong>SO2</strong></code></td>
      <td>Rotations in 2D.</td>
      <td><em>(real, imaginary):</em> unit complex (∈ S<sup>1</sup>)</td>
    </tr>
    <tr>
      <td><code>jaxlie.<strong>SE2</strong></code></td>
      <td>Proper rigid transforms in 2D.</td>
      <td><em>(real, imaginary, x, y):</em> unit complex &amp; translation</td>
    </tr>
    <tr>
      <td><code>jaxlie.<strong>SO3</strong></code></td>
      <td>Rotations in 3D.</td>
      <td><em>(qw, qx, qy, qz):</em> wxyz quaternion (∈ S<sup>3</sup>)</td>
    </tr>
    <tr>
      <td><code>jaxlie.<strong>SE3</strong></code></td>
      <td>Proper rigid transforms in 3D.</td>
      <td><em>(qw, qx, qy, qz, x, y, z):</em> wxyz quaternion &amp; translation</td>
    </tr>
  </tbody>
</table>

Where each group supports:

- Forward- and reverse-mode AD-friendly **`exp()`**, **`log()`**,
  **`adjoint()`**, **`apply()`**, **`multiply()`**, **`inverse()`**,
  **`identity()`**, **`from_matrix()`**, and **`as_matrix()`** operations. (see
  [./examples/se3_example.py](./examples/se3_basics.py))
- Helpers for optimization on manifolds (see
  [./examples/se3_optimization.py](./examples/se3_optimization.py),
  <code>jaxlie.<strong>manifold.\*</strong></code>).
- Compatibility with standard JAX function transformations. (see
  [./examples/vmap_example.py](./examples/vmap_example.py))
- (Un)flattening as pytree nodes.
- Serialization using [flax](https://github.com/google/flax).

We also implement various common utilities for things like uniform random
sampling (**`sample_uniform()`**) and converting from/to Euler angles (in the
`SO3` class).

---

### Install (Python >=3.7)

```bash
# Python 3.6 releases also exist, but are no longer being updated.
pip install jaxlie
```

---

### In the wild

- [jaxfg](https://github.com/brentyi/jaxfg) applies `jaxlie` to nonlinear least
  squares problems with block-sparse structure. (for pose graph optimization,
  bundle adjustment, etc)
- [tensorf-jax](https://github.com/brentyi/tensorf-jax) is an unofficial
  implementation of
  [Tensorial Radiance Fields (Chen et al, ECCV 2022)](https://apchenstu.github.io/TensoRF/)
  using `jaxlie`.
  ![Render of a lego](https://github.com/brentyi/tensorf-jax/raw/main/lego_render.gif)

---

### Misc

`jaxlie` was originally written for our IROS 2021 paper
([link](https://github.com/brentyi/dfgo)). If it's useful for you, you're
welcome to cite:

```
@inproceedings{yi2021iros,
    author={Brent Yi and Michelle Lee and Alina Kloss and Roberto Mart\'in-Mart\'in and Jeannette Bohg},
    title = {Differentiable Factor Graph Optimization for Learning Smoothers},
    year = 2021,
    BOOKTITLE = {2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}
}
```
