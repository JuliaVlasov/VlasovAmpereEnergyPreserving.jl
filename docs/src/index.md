# VlasovAmpereEnergyPreserving.jl

This package explores energy-preserving semi-Lagrangian methods for kinetic
plasma simulations. The examples correspond to the numerical tests in Section 6
of Crouseilles, Liu and Yue, *Semi-Lagrangian SAV method for Vlasov-Maxwell
equations*:

- [Two-stream instability](two-stream.md): Vlasov–Ampère, Section 6.1.
- [Weibel instability](weibel.md): Vlasov–Maxwell, Section 6.2.
- [Landau–Weibel instability](landau-weibel.md): Vlasov–Maxwell, Section 6.3.

Section 6.4 benchmarks a separate JAX implementation using the Weibel and
Landau–Weibel cases; it does not introduce another initial condition. See the
[JAX acceleration benchmark](jax-benchmark.md) for the benchmark setup and
its relationship to these Julia examples.

## Installation

Install Julia (version 1.11 or later is recommended), then clone the repository
and instantiate its dependencies:

```sh
git clone https://github.com/JuliaVlasov/VlasovAmpereEnergyPreserving.jl.git
cd VlasovAmpereEnergyPreserving.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Run the commands on the individual case pages from the repository root.

## Numerical method

The paper introduces scalar auxiliary variables (SAVs) for electric-field
energy and splits the kinetic and field evolution into simpler subproblems.
Spatial transport is treated with a Fourier spectral method; velocity transport
uses a semi-Lagrangian method with high-order Lagrange interpolation. The
splitting is designed to preserve Gauss's law and a modified total energy.
Finite grids, velocity truncation and interpolation can affect those invariants
in a numerical run. The two-stream example uses a symmetric splitting, while
the Vlasov–Maxwell examples use the first-order Lie splitting.

## Build the documentation

From the repository root, create the documentation environment and build:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html` in a browser. Building the documentation does not
run the simulations.

## Documentation deployment

The Documentation workflow builds pull requests without publishing them.
Pushes to `main` publish the development documentation to `gh-pages`, while
version tags such as `v0.1.0` publish versioned documentation and update
`stable` to the latest release. Local builds do not deploy.

In the repository's **Settings > Pages**, select **Deploy from a branch**,
then **gh-pages** and **/ (root)**. The branch is created by the first successful
deployment. The workflow uses the `DOCUMENTER_KEY` secret and its write-enabled
Documenter deploy key; it can also push with `GITHUB_TOKEN`, but pushes made
with that token do not trigger GitHub Pages builds.
