# JAX SAV examples

This directory contains a standalone JAX implementation of the SAV schemes for
the paper's Section 6.1 Vlasov–Ampère two-stream case and Sections 6.2–6.3
VM1d-2v cases. It does not change the Julia package environment.

## Install and run

Use Python 3.10 or newer with [uv](https://docs.astral.sh/uv/):

```sh
cd jax_code
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python VM-SAV-JAX.py --case two-stream
python VM-SAV-JAX.py --case weibel
python VM-SAV-JAX.py --case landau-weibel
```

By default, the script uses the paper's grid sizes, time steps, final times,
perturbations, and Weibel velocity bounds. Results are written to
`results/<case>/diagnostics.csv` and `final_fields.csv`. The requirements file
installs JAX with its CPU backend. Check the detected devices with:

```sh
python -c "import jax; print(jax.devices())"
```

On Linux with an NVIDIA GPU, choose a CUDA wheel compatible with the GPU
architecture. CUDA 12 supports NVIDIA SM 5.2 and newer (including compute
capability 6.1); CUDA 13 requires SM 7.5 or newer. To replace the CPU/default
installation with the CUDA 12 wheel:

```sh
uv pip install --python .venv/bin/python --reinstall 'jax[cuda12]'
```

For newer GPUs with CUDA 13-compatible drivers, use
`uv pip install --python .venv/bin/python --reinstall 'jax[cuda13]'` instead.
Then verify the detected backend and devices:

```sh
.venv/bin/python -c "import jax; print(jax.devices())"
```

Install a JAX wheel supported by your Python version, and ensure the NVIDIA
driver meets the selected CUDA wheel's requirements; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
CUDA is not available on macOS; use the CPU build there.

### Troubleshooting duplicate CUDA plugins

If `jax.devices()` reports `PJRT_Api already exists for device type cuda` but
also prints `[CudaDevice(id=0)]`, JAX can see a GPU, but the environment may
contain duplicate CUDA plugins (for example, leftovers from switching between
CUDA 13 and CUDA 12). Rather than reinstalling over that environment, create a
separate clean CUDA 12 environment:

```sh
uv venv .venv-cuda12
uv pip install --python .venv-cuda12/bin/python -r requirements.txt 'jax[cuda12]'
.venv-cuda12/bin/python -c "import jax; print(jax.devices())"
```

All case parameters can be overridden on the command line. For example:

```sh
python VM-SAV-JAX.py --case weibel --nx 8 --nv 9 --dt 0.0625 --final-time 0.125
python VM-SAV-JAX.py --case landau-weibel --vx-bounds -5 5 --vy-bounds -7 7
```

For Section 6.3, the paper does not state finite velocity limits. This
implementation explicitly chooses `vx ∈ [-6, 6]` and `vy ∈ [-8, 8]` by
default; override them with `--vx-bounds` and `--vy-bounds`. Section 6.1 uses
`v ∈ [-8, 8]` by default and accepts `--v-bounds`. Velocity integrals use
trapezoidal quadrature on endpoint-inclusive grids, and the initial
distribution is normalized on that discrete grid.

The Section 6.1 SAV discrepancy sweep is available with:

```sh
python VM-SAV-JAX.py --case two-stream --sav-sweep
```

It evaluates `dt = 10**(-p)` for `p = 1,…,6` through the selected final time
(default `T = 100`) and writes `sav_sweep.csv`. The smallest time step entails
100 million steps at the default final time; use `--final-time` and/or
`--sweep-exponents` for a shorter or smaller diagnostic sweep.

## Numerical notes

- The scalar auxiliary states are independent unknowns `r` (not `r**2`).
  They are initialized as `sqrt(0.5 * integral(E**2) + sav_shift)` and are
  advanced by the midpoint SAV formulas; they are never reconstructed from
  the electric field during time stepping. A small positive shift (default
  `1e-12`) regularizes the initially zero `Ey` SAV state. Its constant offset
  is removed in the reported SAV-versus-field discrepancy.
- The VM Strang composition follows the paper's ordering: magnetic half-step,
  electric/SAV half-step, full space-transport/Ampère step, electric/SAV
  half-step, magnetic half-step. The 1D Vlasov–Ampère case uses a symmetric
  transport/electric/transport composition.
- Space is periodic and differentiated/transported spectrally. Velocity
  interpolation is tensor-product cubic Lagrange in the interior, bilinear
  near the grid edge, and zero outside the configured bounds.
- Diagnostics include the Gauss-law `L2` residual, kinetic and field energies,
  relative modified-energy error, and each SAV/field-energy discrepancy.
  The SAV shift and finite velocity truncation are numerical assumptions, not
  changes to the paper's governing equations.
