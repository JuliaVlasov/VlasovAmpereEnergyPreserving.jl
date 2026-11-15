# +
@doc raw"""
    HE!(f, r, v, g, nv, kv, dv, nx, dx, dt; interpolate=semilag_d2)

Advance the one-velocity electric SAV subsystem with fixed profile `g`:

```math
\partial_t f+rg\,\partial_vf=0,\qquad
\dot r=-\frac12\int gJ\,dx,\qquad J=\int vf\,dv.
```

Approximate velocity and spatial integrals by uniform-grid sums. Use the
midpoint acceleration ``\overline E=\tfrac12(r^{n+1}+r^n)g`` to translate
each column of `f` with `interpolate(u, speed, dt, dv)`.
`f` has dimensions `(nv, nx)` and is updated in place. Return the new
auxiliary scalar; `g` is unchanged.
"""
function HE!(f, r, v, g, nv, kv, dv, nx, dx, dt; interpolate = semilag_d2)
    J = vec(sum(v .* f; dims = 1)) .* dv
    rho = vec(sum(f; dims = 1)) .* dv
    Jnpdemi = J .+ 0.25 * dt * r * g .* rho
    rnp1 =
        (1 / (1 + (dt * dt / 8) * sum(rho .* g .^ 2) * dx)) * (r - 0.5 * dt * sum(g .* Jnpdemi) * dx)

    enew = 0.5 * (rnp1 + r) * g

    u = zeros(nv)
    for i in eachindex(enew)
        u .= f[:, i]
        u .= interpolate(u, enew[i], dt, dv)
        f[:, i] .= u
    end

    return rnp1
end


export OperatorHE

struct OperatorHE{T}

    mesh::Mesh1D2V{T}
    kx::Vector{T}
    Jx::Vector{T}
    Jy::Vector{T}
    rho::Vector{T}


    function OperatorHE(mesh::Mesh1D2V{T}) where {T}

        kx = collect(2π / (mesh.xmax - mesh.xmin) .* fftfreq(mesh.nx, mesh.nx))
        Jx = zeros(T, mesh.nx)
        Jy = zeros(T, mesh.nx)
        rho = zeros(T, mesh.nx)
        return new{T}(mesh, kx, Jx, Jy, rho)

    end

end


@doc raw"""
    solve!(f, rx, ry, B, gx, gy, he::OperatorHE, dt; interpolate=semilag_d2)

Advance the electric SAV splitting subsystem with fixed profiles `gx`, `gy`:

```math
\begin{aligned}
\partial_t f + r_xg_x\partial_{v_x}f + r_yg_y\partial_{v_y}f &= 0,\\
\partial_t B &= -\partial_x(r_yg_y),\\
\dot r_x &= -\frac12\int g_xJ_x\,dx,\\
\dot r_y &= -\frac12\int g_y(J_y+\partial_x B)\,dx.
\end{aligned}
```

The scalar auxiliary variables (SAVs) are `rx`, `ry`. The velocity moments are
``\rho=\int f\,dv_x\,dv_y`` and ``J_\alpha=\int v_\alpha f\,dv_x\,dv_y``
for ``\alpha\in\{x,y\}``, approximated by uniform-grid sums.

Use the midpoint accelerations
``\overline E_\alpha=\tfrac12(r_\alpha^{n+1}+r_\alpha^n)g_\alpha``
to translate `f` in each velocity direction. `interpolate(u, speed, dt, dv)`
performs each one-dimensional semi-Lagrangian translation. Spatial
derivatives are periodic Fourier derivatives.

Update `f` and `B` in place and return `(rxnp1, rynp1)`. The supplied profiles
are unchanged; the electric-field arrays belonging to other splitting
subsystems are not advanced here.
"""
function solve!(f, rx, ry, B, gx, gy, he::OperatorHE, dt; interpolate = semilag_d2)

    nvx, nvy, nx = size(f)

    dx = he.mesh.dx
    dvx = he.mesh.dvx
    dvy = he.mesh.dvy

    he.Jx .= vec(sum(he.mesh.vx .* f; dims = [1, 2])) .* dvx .* dvy
    he.Jy .= vec(sum(he.mesh.vy' .* f; dims = [1, 2])) .* dvx .* dvy
    he.rho .= vec(sum(f; dims = [1, 2])) .* dvx .* dvy

    Jxnpdemi = he.Jx + 0.25dt * rx * gx .* he.rho

    denom = 1 + (dt * dt / 8) * sum(he.rho .* gx .^ 2) * dx

    rxnp1 = (rx - 0.5dt * sum(gx .* he.Jx) * dx - (dt * dt / 8) * rx * sum(he.rho .* gx .^ 2) * dx) / denom

    Exnew = 0.5 * (rxnp1 + rx) * gx

    Jynpdemi = he.Jy .+ 0.25dt * ry * gy .* he.rho
    dxgy = real(ifft(1im * he.kx .* fft(gy)))
    dx2gy = real(ifft(-he.kx .* he.kx .* fft(gy)))
    dxB = real(ifft(1im * he.kx .* fft(B)))
    denom = 1 + (dt * dt / 8) * sum(he.rho .* gy .^ 2) * dx + (dt * dt / 8) * sum(dxgy .* dxgy) * dx

    rynp1 = ry - 0.5 * dt * sum(he.Jy .* gy) * dx
    rynp1 += 0.5dt * sum(B .* dxgy) * dx
    rynp1 -= (dt * dt / 8) * ry * sum(he.rho .* gy .^ 2) * dx
    rynp1 -= (dt * dt / 8) * ry * sum(dxgy .* dxgy) * dx

    rynp1 /= denom

    Eynew = 0.5 * (rynp1 + ry) * gy

    faraday = real(ifft(1im * he.kx .* fft(Eynew)))

    B .= B .- dt * faraday

    fvx = zeros(nvx)
    for j in axes(f, 2), i in eachindex(Exnew)
        fvx .= f[:, j, i]
        f[:, j, i] .= interpolate(fvx, Exnew[i], dt, dvx)
    end

    fvy = zeros(nvy)
    for j in axes(f, 1), i in eachindex(Exnew)
        fvy .= f[j, :, i]
        f[j, :, i] .= interpolate(fvy, Eynew[i], dt, dvy)
    end

    return rxnp1, rynp1
end

function HE(
        fn::Array{Float64, 3},
        vx::Vector{Float64},
        vy::Vector{Float64},
        Vx::Matrix{Float64},
        Vy::Matrix{Float64},
        rx::Float64,
        gx::Vector{Float64},
        ry::Float64,
        gy::Vector{Float64},
        B::Vector{Float64},
        Nvx::Int,
        Nvy::Int,
        kk::Vector{Float64},
        kv::Matrix{Float64},
        dvx::Float64,
        dvy::Float64,
        Nx::Int,
        dx::Float64,
        dt::Float64,
        d::Int,
    )

    # Pre-allocate arrays
    Jx = zeros(Nx)
    Jy = zeros(Nx)
    rho = zeros(Nx)

    # Compute moments (optimized with views and reduced allocations)
    @inbounds for i in 1:Nx
        fn_slice = view(fn, :, :, i)
        Jx[i] = sum(vx .* sum(fn_slice, dims = 2)) * dvx * dvy
        Jy[i] = sum(vy' .* sum(fn_slice, dims = 1)) * dvx * dvy
        rho[i] = sum(fn_slice) * dvx * dvy
    end

    # Update rx and Ex
    Jxnpdemi = Jx .+ 0.25 .* dt .* rx .* gx .* rho
    denom = 1 + (dt^2 / 8) * sum(rho .* gx .^ 2) * dx
    rxnp1 =
        (rx - 0.5 * dt * sum(gx .* Jx) * dx - (dt^2 / 8) * rx * sum(rho .* gx .^ 2) * dx) /
        denom
    Exnew = 0.5 .* (rxnp1 .+ rx) .* gx

    # Update ry and Ey
    Jynpdemi = Jy .+ 0.25 .* dt .* ry .* gy .* rho
    gy_hat = fft(gy)
    dxgy = real(ifft(im .* kk .* gy_hat))
    dx2gy = real(ifft(-kk .^ 2 .* gy_hat))
    dxB = real(ifft(im .* kk .* fft(B)))

    denom = 1 + (dt^2 / 8) * sum(rho .* gy .^ 2) * dx + (dt^2 / 8) * sum(dxgy .^ 2) * dx
    rynp1 =
        (
        ry - 0.5 * dt * sum(Jy .* gy) * dx + 0.5 * dt * sum(B .* dxgy) * dx -
            (dt^2 / 8) * ry * sum(rho .* gy .^ 2) * dx -
            (dt^2 / 8) * ry * sum(dxgy .^ 2) * dx
    ) / denom
    Eynew = 0.5 .* (rynp1 .+ ry) .* gy

    # Update B field
    faraday = real(ifft(im .* kk .* fft(Eynew)))
    B .-= dt .* faraday

    # Semi-Lagrangian steps (optimized with threading)
    fn_temp = similar(fn)

    # x-direction
    @inbounds Threads.@threads for j in 1:Nvy
        for i in 1:Nx
            ft = view(fn, :, j, i)
            fn_temp[:, j, i] = semilag(ft, Nvx, Exnew[i], dt, dvx, d)
        end
    end
    fn .= fn_temp

    # y-direction
    @inbounds Threads.@threads for j in 1:Nvx
        for i in 1:Nx
            ft = view(fn, j, :, i)
            fn_temp[j, :, i] = semilag(ft, Nvy, Eynew[i], dt, dvy, d)
        end
    end
    fn .= fn_temp

    return fn, rxnp1, rynp1, B
end
