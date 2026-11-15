export compute_diag

@doc raw"""
    compute_diag(f, Ex, Ey, B, rx, ry, mesh::Mesh1D2V, t)

Return a dictionary of mass, field norms, and energy diagnostics at time `t`.
All integrals use uniform-grid sums over `mesh`:

```math
\begin{aligned}
K &= \frac12\int (v_x^2+v_y^2)f\,dx\,dv_x\,dv_y,\\
M &= \int f\,dx\,dv_x\,dv_y,\\
\mathcal H_{\mathrm{SAV}} &= K+r_x^2+r_y^2+\frac12\int B^2\,dx,\\
\mathcal H_{\mathrm{physical}} &= K+\frac12\int(E_x^2+E_y^2+B^2)\,dx.
\end{aligned}
```

`KineticEnergy` and `IntegralF` contain ``K`` and ``M``.
`TotalEnergy1` and `TotalEnergy2` contain the SAV and physical energies.
`BNorm`, `ExNorm`, and `EyNorm` are squared spatial norms (without a factor
of one half); `RxSquare`, `RySquare`, and `time` contain `rx^2`, `ry^2`, and `t`.
No input arrays are modified.
"""
function compute_diag(f, Ex, Ey, B, rx, ry, mesh::Mesh1D2V, t)

    vx = mesh.vx
    vy = mesh.vy
    dx = mesh.dx
    dvx = mesh.dvx
    dvy = mesh.dvy
    dxv = dx * dvx * dvy
    kin = 0

    for k in axes(f, 3), i in eachindex(vx), j in eachindex(vy)
        kin += (vx[i] * vx[i] + vy[j] * vy[j]) * f[i, j, k]
    end

    kin = 0.5 * kin * dxv

    diag = Float64[]
    push!(diag, t)
    push!(diag, dx * sum(B .* B))
    push!(diag, dx * sum(Ex .* Ex))
    push!(diag, dx * sum(Ey .* Ey))
    push!(diag, kin)
    push!(diag, sum(f) * dxv)
    push!(diag, kin + rx^2 + ry^2 + 0.5 * dx * sum(B .* B))
    push!(diag, rx^2)
    push!(diag, ry^2)
    push!(diag, kin + 0.5dx * sum(Ex .* Ex) + 0.5dx * sum(Ey .* Ey) + 0.5dx * sum(B .* B))

    names = (
        "time",
        "BNorm",
        "ExNorm",
        "EyNorm",
        "KineticEnergy",
        "IntegralF",
        "TotalEnergy1",
        "RxSquare",
        "RySquare",
        "TotalEnergy2",
    )
    return Dict(zip(names, diag))
end
