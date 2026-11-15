# JAX acceleration benchmark

Section 6.4 benchmarks a separate JAX/XLA implementation of the
one-space, two-velocity Vlasov–Maxwell SAV solver. It reuses the initial
conditions of the [Weibel](weibel.md) and [Landau–Weibel](landau-weibel.md)
tests, rather than introducing a new physical case.

The paper reports runs on an NVIDIA Quadro RTX 6000 GPU using JAX's
single-precision arithmetic. Its JAX solver retains Fourier spatial transport,
but uses JAX's linear interpolator for the second splitting step and a custom
JIT-compiled two-dimensional cubic B-spline interpolation for the third step.
These interpolation choices differ from the Julia examples.

## Timing experiments

Each timing is an average over 100 runs:

| Case | Final time | ``N_x \times N_{v_x} \times N_{v_y}`` | Time steps |
|:--|:--|:--|:--|
| Weibel | 300 | ``8\times23\times23``, ``16\times47\times47``, ``32\times95\times95`` | 0.12500, 0.06250, 0.03125, 0.01562 |
| Landau–Weibel | 50 | ``23^3``, ``47^3``, ``95^3``, ``191^3`` | 0.25000, 0.12500, 0.06250, 0.03125 |

Section 6.4 reports approximately linear runtime scaling with the number of
time steps. The timings are specific to its JAX implementation and hardware;
they are not Julia benchmarks.

The paper also presents a refined Weibel run with a ``64\times191\times191``
grid and ``\Delta t=0.01562``. Figures 6 and 7 show field energies, the
modified total energy and its change, and velocity distributions at four
spatial locations at ``t=300``.

## Relation to the Julia examples

The Julia examples run the physical cases with the default parameters from
Sections 6.2 and 6.3:

```sh
julia --project examples/weibel-instability.jl
julia --project examples/landau-weibel-instability.jl
```

## Julia CUDA comparison

The optional `benchmarks/cuda` environment implements the same physical cases
on NVIDIA GPUs in single precision. Its default `paper` mode uses one-dimensional
linear interpolation for electric transport and two-dimensional cubic B-spline
interpolation for magnetic transport. Linear interpolation clamps at velocity
endpoints, like `jax.numpy.interp` with its default arguments. The spline uses
periodic velocity boundaries and Fourier coefficient prefiltering.

The paper does not specify the custom JAX B-spline's boundary treatment,
coefficient computation, or implementation; its source is not included here.
Consequently, identical numerical results or kernel costs cannot be assumed.
Like the Julia examples, this implementation uses shifted electric SAV energies
and assumes ``v_x\in[-6,6)``, ``v_y\in[-8,8)`` for Landau–Weibel, whose
velocity bounds are unspecified in Section 6.3.

On a machine with a supported NVIDIA GPU and CUDA driver, from the repository
root:

```sh
julia --project=benchmarks/cuda -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=benchmarks/cuda benchmarks/cuda/test.jl
julia --project=benchmarks/cuda benchmarks/cuda/run.jl weibel 16 47 0.0625
julia --project=benchmarks/cuda benchmarks/cuda/run.jl landau-weibel 47 47 0.125
```

The arguments are case, spatial grid size, each velocity grid size, time
step, and optionally the number of runs (default 100) and interpolation method
(`paper` or `high_order`). The latter selects the original Julia CPU example's
degree-19 interpolation and is **not** interpolation-matched to JAX. For example:

```sh
julia --project=benchmarks/cuda benchmarks/cuda/run.jl weibel 16 47 0.0625 100 high_order
```

Use the grid sizes and time steps in the table above to cover the paper's timing configurations. The
printed time step 0.01562 in the paper is rounded: use `0.015625` for an
integer number of steps to the stated final time.

The runner compiles a one-step warm-up, then reports mean wall-clock time for
100 synchronized, fresh full simulations, including initialization and device
transfers. The paper does not specify those timing boundaries. A separate
diagnostic run prints CSV histories of the squared magnetic and electric field
norms, modified energy and Gauss-law residual. For a final distribution, use
`simulate_vm` from `benchmarks/cuda/solver.jl` with `array_type = CuArray` and
read its `f` result. The paper's JAX source, timing tables and plotting code are
not part of this package.

The CUDA path requires validation on NVIDIA hardware. CPU backend tests check
the interpolation operators and short-time agreement with the Julia examples.
This implementation does **not** reproduce the paper's reported conservation;
do not interpret its timings as a validated JAX performance comparison.

### Conservation investigation

The diagnostic uses the paper's modified energy: kinetic energy plus the
squares of both electric SAVs and half the magnetic-field squared norm. The
Julia implementation initializes each squared SAV with an extra 1; subtracting
2 from the reported energy removes that constant shift. The Gauss diagnostic
uses the full discrete norm of ``\partial_x E_x-(\rho-1)``.

For a complete Weibel run on the CPU backend with ``N_x=16``, ``N_v=47`` and
``\Delta t=1/16``, the `paper` mode's modified energy increased from 0.006535
by 0.001236 in Float64 (about 19%); Float32 increased it by 0.001316 (about
20%). Tracking the splitting steps in Float64 attributes 0.001235649 of the
increase to electric transport and about 0.0000000024 to magnetic transport.
The predicted second-moment diffusion from linear interpolation is 0.001235647.
Linear interpolation of a fractional velocity-cell shift adds a positive
second-moment error; the SAV update assumes the moments of an exact
translation. The high-order Julia mode decreased the energy by approximately
``4.9\times10^{-9}`` in the same Float64 run.

The initial Float64 Gauss residual is negligible. At ``t=300`` the `paper`
mode's Float64 residual is 0.013984, almost entirely the spatial Nyquist mode
on the even 16-point grid. A real collocation derivative cannot represent that
mode. Repeating with odd ``N_x=15`` or 17 reduces the final residual to
``7.3\times10^{-7}`` or ``5.0\times10^{-7}``, respectively, but changes the
paper's grid and does not remove the energy drift. The Float32 final residual
is 0.013263; its non-Nyquist components are larger than in Float64.

The paper does not disclose enough of its JAX spline implementation and
boundary treatment to determine why its published conservation behavior differs
from this implementation. Neither these CPU results nor the untested GPU path
establish equivalence to the JAX code.
