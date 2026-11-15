struct OperatorHE2{T}

    mesh::Mesh1D2V{T}

    OperatorHE2(mesh::Mesh1D2V{T}) where {T} = new{T}(mesh)

end
@doc raw"""
    solve!(f, Ex, Ey, gx, gy, rx, ry, he2, dt)

Advance the electric SAV subsystem without magnetic-field coupling:

```math
\begin{aligned}
\partial_t f+r_xg_x\partial_{v_x}f+r_yg_y\partial_{v_y}f &= 0,\\
\dot r_x &= -\frac12\int g_xJ_x\,dx,\qquad
\dot r_y=-\frac12\int g_yJ_y\,dx,\\
\partial_t E_x = \partial_t E_y = \partial_t B &= 0.
\end{aligned}
```

Here ``J_\alpha=\int v_\alpha f\,dv_x\,dv_y``. `he2` supplies the mesh.
First recompute `gx` and `gy` from the fixed fields and auxiliary scalars
with `updateg!`. Translate `f` in both velocity directions using
``\overline E_\alpha=\tfrac12(r_\alpha^{n+1}+r_\alpha^n)g_\alpha``
and second-degree semi-Lagrangian interpolation.

Update `f`, `gx`, and `gy` in place and return `(rxnp1, rynp1)`.
`Ex` and `Ey` are unchanged.
"""
function solve!(f, Ex, Ey, gx, gy, rx, ry, he2, dt)

    nvx, nvy, nx = size(f)

    dx = he2.mesh.dx
    dvx = he2.mesh.dvx
    dvy = he2.mesh.dvy

    Jx = zeros(nx)
    Jy = zeros(nx)
    rho = zeros(nx)

    for i in 1:nx
        Jx[i] = 0.0
        Jy[i] = 0.0
        rho[i] = 0.0
        for j in 1:nvx, k in 1:nvy
            Jx[i] += he2.mesh.vx[j] * f[j, k, i]
            Jy[i] += he2.mesh.vy[k] * f[j, k, i]
            rho[i] += f[j, k, i]
        end
    end

    Jx .*= dvx * dvy
    Jy .*= dvx * dvy
    rho .*= dvx * dvy

    updateg!(gx, gy, Ex, Ey, rx, ry)

    denom = 1 + (dt * dt / 8.0) * sum(rho .* gx .^ 2) * dx
    rxnp1 = (rx - 0.5 * dt * sum(gx .* Jx) * dx - (dt * dt / 8.0) * rx * sum(rho .* gx .^ 2) * dx) / denom
    rxdemi = rx / denom - (0.25 * dt / denom) * sum(gx .* Jx) * dx
    Exnew = rxdemi * gx

    denom = 1 + (dt * dt / 8.0) * sum(rho .* gy .^ 2) * dx
    rynp1 = (ry - 0.5 * dt * sum(Jy .* gy) * dx - (dt * dt / 8.0) * ry * sum(rho .* gy .^ 2) * dx) / denom
    rydemi = ry / denom - (0.25 * dt / denom) * sum(gy .* Jy) * dx
    Eynew = rydemi * gy

    fvx = zeros(nvx)
    for j in 1:nvy, i in 1:nx
        fvx .= f[:, j, i]
        fvx .= semilag_d2(fvx, Exnew[i], dt, dvx)
        f[:, j, i] .= fvx
    end

    fvy = zeros(nvy)
    for j in 1:nvx, i in 1:nx
        fvy .= f[j, :, i]
        fvy .= semilag_d2(fvy, Eynew[i], dt, dvy)
        f[j, :, i] .= fvy
    end

    return rxnp1, rynp1
end
