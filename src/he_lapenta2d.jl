struct OperatorLapentaHE{T}

    mesh::Mesh1D2V
    Jx::Vector{T}
    Jy::Vector{T}
    rho::Vector{T}
    Ex::Vector{T}
    Ey::Vector{T}
    fvx::Vector{T}
    fvy::Vector{T}

    function OperatorLapentaHE(mesh::Mesh1D2V{T}) where {T}

        Jx = zeros(T, mesh.nx)
        Jy = zeros(T, mesh.nx)
        rho = zeros(T, mesh.nx)
        Ex = zeros(T, mesh.nx)
        Ey = zeros(T, mesh.nx)
        fvx = zeros(T, mesh.nvx)
        fvy = zeros(T, mesh.nvy)

        return new{T}(mesh, Jx, Jy, rho, Ex, Ey, fvx, fvy)


    end


end

@doc raw"""
    solve!(f, Ex, Ey, he::OperatorLapentaHE, dt)

Advance the electric acceleration and current subsystem:

```math
\begin{aligned}
\partial_t f+E_x\partial_{v_x}f+E_y\partial_{v_y}f &= 0,\\
\partial_t E_x &= -J_x,\qquad \partial_t E_y=-J_y,\\
\partial_t B &= 0.
\end{aligned}
```

The moments are ``\rho=\int f\,dv_x\,dv_y`` and
``J_\alpha=\int v_\alpha f\,dv_x\,dv_y``. The field midpoint used for
semi-Lagrangian translation in each velocity direction is

```math
\overline E_\alpha
=\frac{E_\alpha^n-\frac{\Delta t}{2}J_\alpha^n}
       {1+\frac{\Delta t^2}{4}\rho^n},
\qquad E_\alpha^{n+1}=2\overline E_\alpha-E_\alpha^n.
```

`he` stores the moment and interpolation work arrays.
Update `f`, `Ex`, and `Ey` in place and return `nothing`.
"""
function solve!(f, Ex, Ey, he::OperatorLapentaHE, dt)

    nvx, nvy, nx = size(f)

    dvx = he.mesh.dvx
    dvy = he.mesh.dvy

    for ix in axes(f, 3), ivy in axes(f, 2), ivx in axes(f, 1)
        he.Jx[ix] += he.mesh.vx[ivx] * f[ivx, ivy, ix]
        he.Jy[ix] += he.mesh.vy[ivy] * f[ivx, ivy, ix]
        he.rho[ix] += f[ivx, ivy, ix]
    end

    he.Jx .*= dvx * dvy
    he.Jy .*= dvx * dvy
    he.rho .*= dvx * dvy

    dt2_rho4 = (dt^2) .* he.rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2

    he.Ex .= (1 ./ (1 .+ dt2_rho4)) .* Ex .- (dt ./ (2 .+ dt2_rho2)) .* he.Jx
    he.Ey .= (1 ./ (1 .+ dt2_rho4)) .* Ey .- (dt ./ (2 .+ dt2_rho2)) .* he.Jy

    for i in eachindex(he.Ex), j in axes(f, 2)
        he.fvx .= f[:, j, i]
        he.fvx .= semilag_d2(he.fvx, he.Ex[i], dt, dvx)
        f[:, j, i] .= he.fvx
    end

    for i in eachindex(he.Ey), k in axes(f, 1)
        he.fvy .= f[k, :, i]
        he.fvy .= semilag_d2(he.fvy, he.Ey[i], dt, dvy)
        f[k, :, i] .= he.fvy
    end

    Ex .= 2 .* he.Ex .- Ex
    Ey .= 2 .* he.Ey .- Ey

    return

end
