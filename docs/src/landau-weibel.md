# Landau–Weibel instability

Section 6.3 combines longitudinal Landau damping with a transverse Weibel
instability. The initial distribution and fields are

```math
f_0(x,v_x,v_y) =
\frac{\exp[-(v_x^2/\sigma_1^2+v_y^2/\sigma_2^2)/2]}
     {2\pi\sigma_1\sigma_2}
\bigl(1+\alpha\cos(kx)\bigr),\quad
B_0=\beta\cos(kx),\quad E_{y,0}=0.
```

The initial longitudinal field is obtained from Poisson's equation.

| Parameter | Paper Section 6.3 |
|:--|:--|
| ``\sigma_1, \sigma_2`` | ``1, \sqrt{2}`` |
| ``k, \alpha, \beta`` | ``0.5, 10^{-3}, -10^{-6}`` |
| Space domain | ``[0,2\pi/k)`` |
| Grid | ``N_x=N_{v_x}=N_{v_y}=47`` |
| Time | ``\Delta t=0.125``, ``T=50`` |

The paper reports a longitudinal electric-field damping rate of approximately
``-0.153`` and growth of the transverse fields.

## Run the Julia example

```sh
julia --project examples/landau-weibel-instability.jl
```

The example uses the same SAV solver as the [Weibel case](weibel.md). Section
6.3 does not specify finite velocity bounds, so this example assumes periodic
grids ``v_x\in[-6,6)`` and ``v_y\in[-8,8)``. The finite longitudinal
velocity resolution produces recurrence near the end of the run. The example
does not reproduce the paper's ECSL and VM-split comparisons.

```julia
include("examples/landau-weibel-instability.jl")
result = run_landau_weibel()
result.time
result.ex_norm  # squared L2 norm
result.ey_norm
result.magnetic_norm
result.total_energy
result.gauss_error
```
