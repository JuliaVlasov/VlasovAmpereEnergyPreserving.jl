struct OperatorHB2{T}

    mesh::Mesh1D2V{T}
    kx::Vector{T}
    kvx::Vector{T}
    kvy::Vector{T}
    ft::Array{Complex{T}, 3}

    function OperatorHB2(mesh::Mesh1D2V{T}) where {T}

        kx = collect(2π / (mesh.xmax - mesh.xmin) * fftfreq(mesh.nx, mesh.nx))
        kvx = collect(2π / (mesh.vxmax - mesh.vxmin) * fftfreq(mesh.nvx, mesh.nvx))
        kvy = collect(2π / (mesh.vymax - mesh.vymin) * fftfreq(mesh.nvy, mesh.nvy))

        ft = zeros(Complex{T}, (mesh.nvx, mesh.nvy, mesh.nx))

        return new{T}(mesh, kx, kvx, kvy, ft)

    end


end


@doc raw"""
    solve!(f, Ey, B, gx, gy, rx, ry, hb2::OperatorHB2, dt)

Advance the magnetic SAV subsystem with fixed profile `gy`:

```math
\begin{aligned}
\partial_t f+B\left(v_y\partial_{v_x}f-v_x\partial_{v_y}f\right) &= 0,\\
\partial_t B &= -r_y\partial_x g_y,\qquad
\partial_t E_y=-\partial_x B,\\
\dot r_y &= -\frac12\int g_y\partial_xB\,dx
           =\frac12\int B\partial_xg_y\,dx,\\
\dot r_x &= 0.
\end{aligned}
```

The integration-by-parts identity assumes periodic spatial boundaries.
Use centered periodic differences for the spatial derivatives and midpoint
values of `ry` and `B` for the coupled update. Rotate velocities with three
Fourier shears using the midpoint magnetic field.

Update `f`, `Ey`, and `B` in place and return `(rx, rynp1)`.
`gx`, `gy`, and `rx` are unchanged.
"""
function solve!(f, Ey, B, gx, gy, rx, ry, hb2::OperatorHB2, dt)

    nvx, nvy, nx = size(f)

    dx = hb2.mesh.dx
    hdt = 0.5dt

    dxgy = zeros(nx)

    for i in 2:(nx - 1)
        dxgy[i] = (gy[i + 1] - gy[i - 1]) / (2dx)
    end
    dxgy[1] = (gy[2] - gy[nx]) / (2dx)
    dxgy[nx] = (gy[1] - gy[nx - 1]) / (2dx)

    denom = 1 + (dt * dt / 8.0) * sum(dxgy .* dxgy) * dx
    rynp1 = (ry + 0.5 * dt * sum(B .* dxgy) * dx - (dt * dt / 8.0) * ry * sum(dxgy .* dxgy) * dx) / denom
    rydemi = ry / denom + 0.25 * dt * sum(B .* dxgy) * dx / denom

    Bnp1 = B - dt * rydemi * dxgy
    Bdemi = B - 0.5 * dt * rydemi * dxgy

    dxBdemi = zeros(nx)
    for i in 2:(nx - 1)
        dxBdemi[i] = (Bdemi[i + 1] - Bdemi[i - 1]) / (2dx)
    end


    dxBdemi[1] = (Bdemi[2] - Bdemi[nx]) / (2dx)
    dxBdemi[nx] = (Bdemi[1] - Bdemi[nx - 1]) / (2dx)

    Eynp1 = Ey - dt * dxBdemi

    hb2.ft .= fft(f, 1)
    for l in 1:nx, j in 1:nvy, i in 1:nvx
        hb2.ft[i, j, l] *= exp(-1im * hb2.kvx[i] * hb2.mesh.vy[j] * tan(Bdemi[l] * hdt))
    end
    hb2.ft .= ifft(hb2.ft, 1)

    hb2.ft .= fft(hb2.ft, 2)
    for l in 1:nx, i in 1:nvx, j in 1:nvy
        hb2.ft[i, j, l] *= exp(1im * hb2.kvy[j] * hb2.mesh.vx[i] * sin(Bdemi[l] * dt))
    end
    hb2.ft .= ifft(hb2.ft, 2)

    hb2.ft .= fft(hb2.ft, 1)
    for l in 1:nx, j in 1:nvy, i in 1:nvx
        hb2.ft[i, j, l] *= exp(-1im * hb2.kvx[i] * hb2.mesh.vy[j] * tan(Bdemi[l] * hdt))
    end
    hb2.ft .= ifft(hb2.ft, 1)

    f .= real(hb2.ft)
    Ey .= Eynp1
    B .= Bnp1

    return rx, rynp1

end
