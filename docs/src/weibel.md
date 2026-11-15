# Weibel instability

Section 6.2 considers the one-space, two-velocity Vlasov–Maxwell system. The
initial distribution is a spatially uniform anisotropic Maxwellian:

```math
f_0(x,v_x,v_y) =
\frac{1}{2\pi\sigma_1\sigma_2}
\exp\left[-\frac12\left(\frac{v_x^2}{\sigma_1^2}+
\frac{v_y^2}{\sigma_2^2}\right)\right].
```

The fields start with ``B=\beta\cos(kx)``, ``E_y=0``, and ``E_x``
determined by Poisson's equation.

| Parameter | Paper Section 6.2 |
|:--|:--|
| ``\sigma_1, \sigma_2`` | ``0.02/\sqrt{2}, \sqrt{12}\sigma_1`` |
| ``k, \beta`` | ``1.25, -10^{-4}`` |
| Space domain | ``[0,2\pi/k)`` |
| Velocity domain | ``[-0.2,0.2]\times[-0.5,0.5]`` |
| Grid | ``N_x=16``, ``N_{v_x}=N_{v_y}=47`` |
| Time | ``\Delta t=1/16``, ``T=300`` |

The paper reports a theoretical field growth rate of 0.02784 and an
approximately 0.054 growth rate for the squared magnetic-field norm.

## Run the Julia example

```sh
julia --project examples/weibel-instability.jl
```

The example runs the SAV scheme with degree-19 interpolation; it does not run
the ECSL and VM-split comparisons. Periodic grids omit their upper endpoints.
Each electric SAV energy is shifted by 1 to handle zero initial electric
fields. On the even spatial grid, the real Nyquist mode can contribute to
the Gauss-law residual.

```julia
include("examples/weibel-instability.jl")
result = run_weibel()
result.time
result.magnetic_norm  # squared L2 norm
result.ex_norm
result.ey_norm
result.total_energy
result.gauss_error
```
