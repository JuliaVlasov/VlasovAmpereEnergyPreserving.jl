# Semi-Lagrangian SAV method for Vlasov-Maxwell equations

## Abstract

We introduce a semi-Lagrangian Scalar Auxiliary Variable (SAV) method for the
Vlasov-Maxwell equations. A reformulation with an auxiliary scalar and a
structure-aware time splitting preserves Gauss's law and a modified total
energy while allowing explicit updates without a restrictive CFL condition.
Numerical experiments on the Vlasov-Ampere and one-dimensional, two-velocity
Vlasov-Maxwell models demonstrate the conservation properties and performance
of the method.

[![Build Status](https://github.com/JuliaVlasov/VlasovAmpereEnergyPreserving.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaVlasov/VlasovAmpereEnergyPreserving.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliavlasov.github.io/VlasovAmpereEnergyPreserving.jl/dev/)
[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliavlasov.github.io/VlasovAmpereEnergyPreserving.jl/stable/)

For installation instructions, numerical method, and the paper's test cases,
see the [development documentation](https://juliavlasov.github.io/VlasovAmpereEnergyPreserving.jl/dev/).
Stable documentation becomes available after the first version-tagged release.

## Citation

Crouseilles, N., Liu, H., and Yue, Y. “Semi-Lagrangian SAV method for
Vlasov-Maxwell equations.” *Journal of Computational Physics* 549 (2026),
114606. [doi:10.1016/j.jcp.2025.114606](https://doi.org/10.1016/j.jcp.2025.114606).

```
git clone https://github.com/JuliaVlasov/VlasovAmpereEnergyPreserving.jl
cd VlasovAmpereEnergyPreserving.jl
julia --project
julia> import Pkg; Pkg.instantiate()
julia> include("examples/vlasov-ampere-sav.jl")
```


```
julia> include("examples/simulations2d.jl")
```
