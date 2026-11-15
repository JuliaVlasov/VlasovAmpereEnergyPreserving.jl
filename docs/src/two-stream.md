# Two-stream instability

Section 6.1 of the paper considers the one-dimensional Vlasov–Ampère system
with the initial distribution

```math
f_0(x,v) = \frac{v^2 e^{-v^2/2}}{\sqrt{2\pi}}
           \bigl(1+\alpha\cos(kx)\bigr).
```

The paper uses ``k=0.5``, ``\alpha=0.05``, ``x\in[0,2\pi/k)``,
``N_x=N_v=128``, ``\Delta t=0.5`` and the final time ``T=100``. It uses
Fourier spatial discretization and degree-19 semi-Lagrangian interpolation
(``d=9``). The paper compares VASAV, VA and Vlasov–Poisson schemes and
studies Gauss-law and energy errors.

## Run the Julia example

```sh
julia --project examples/two-stream-instability.jl
```

The paper does not specify a finite velocity interval; the example uses the
periodic grid ``v\in[-6,6)``. It runs
only the SAV scheme, not the paper's VA and Vlasov–Poisson comparison or its
time-step convergence study. The finite velocity interval and the even spatial
grid contribute to nonzero discrete conservation errors. To inspect the
returned histories:

```julia
include("examples/two-stream-instability.jl")
result = run_two_stream()
result.time
result.electric_energy
result.total_energy
result.poisson_error
```

`total_energy` uses the SAV energy; `poisson_error` is the discrete L2 residual
of Gauss's law.
