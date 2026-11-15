export HE_lapenta!

@doc raw"""
    HE_lapenta!(f, E, v, nv, kv, dv, nx, dx, dt)

Advance the one-velocity electric acceleration and current subsystem:

```math
\partial_t f+E\partial_vf=0,\qquad
\partial_t E=-J,\qquad
\rho=\int f\,dv,\quad J=\int vf\,dv.
```

Use uniform-grid velocity moments and the midpoint field

```math
\overline E=\frac{E^n-\frac{\Delta t}{2}J^n}
                  {1+\frac{\Delta t^2}{4}\rho^n},
\qquad
E^{n+1}=E^n-\Delta t\left(J^n+\frac{\Delta t}{2}\rho^n\overline E\right).
```

Translate each column of `f`, of size `(nv, nx)`, with second-degree
semi-Lagrangian interpolation at acceleration ``\overline E``.
Update `f` and `E` in place and return `E`.
"""
function HE_lapenta!(
        f::Matrix{T},
        E::Vector{T},
        v::Vector{T},
        nv::Int,
        kv::Vector{T},
        dv::T,
        nx::Int,
        dx::T,
        dt::T,
    ) where {T <: AbstractFloat}

    # Pre-allocate outputs
    fn1 = similar(f)

    J = vec(sum(v .* f; dims = 1)) .* dv
    rho = vec(sum(f; dims = 1)) .* dv

    dt2_rho4 = (dt^2) .* rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2

    Enew = E ./ (1 .+ dt2_rho4) .- (dt * J) ./ (2 .+ dt2_rho2)

    # Update current and electric field
    Jnpdemi = J .+ 0.5 * dt .* Enew .* rho
    Enp1 = E .- dt .* Jnpdemi

    # Semi-Lagrangian step
    for i in eachindex(Enew)
        ft = f[:, i]
        u1 = semilag_d2(ft, Enew[i], dt, dv)
        f[:, i] .= u1
    end

    return E .= Enp1
end
