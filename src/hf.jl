struct OperatorHF{T}

    mesh::AbstractMesh
    kx::Vector{T}
    kmod::Vector{T}
    ft::Matrix{Complex{T}}
    rhoh::Vector{T}
    rhohvy::Vector{T}
    rhon::Vector{T}
    rhonvy::Vector{T}
    rhs::Vector{Complex{T}}
    Ef::Vector{Complex{T}}
    fnsvy::Array{T, 3}
    fnp1vy::Array{T, 3}

    function OperatorHF(mesh::Mesh1D2V{T}) where {T}

        kx = collect(2π / (mesh.xmax - mesh.xmin) .* fftfreq(mesh.nx, mesh.nx))
        kmod = copy(kx)
        kmod[1] = 1
        ft = zeros(Complex{T}, (mesh.nvx, mesh.nx))
        rhoh = zeros(mesh.nx)
        rhohvy = zeros(mesh.nx)
        rhon = zeros(mesh.nx)
        rhonvy = zeros(mesh.nx)
        rhs = zeros(Complex{T}, mesh.nx)
        Ef = zeros(Complex{T}, mesh.nx)

        fnsvy = zeros(T, (mesh.nvx, mesh.nvy, mesh.nx))
        fnp1vy = zeros(T, (mesh.nvx, mesh.nvy, mesh.nx))

        return new{T}(mesh, kx, kmod, ft, rhoh, rhohvy, rhon, rhonvy, rhs, Ef, fnsvy, fnp1vy)

    end

end

@doc raw"""
    solve!(f, Ex, Ey, hf::OperatorHF, dt)

Advance free streaming and its electric-current coupling:

```math
\begin{aligned}
\partial_t f+v_x\partial_xf &= 0,\\
\partial_t E_x &= -J_x,\qquad \partial_t E_y=-J_y,\\
\partial_t B = \dot r_x = \dot r_y &= 0.
\end{aligned}
```

Here ``J_\alpha=\int v_\alpha f\,dv_x\,dv_y``. Free streaming is exact
for each resolved spatial Fourier mode:

```math
\widehat f^{\,n+1}(k,v_x,v_y)
=e^{-ikv_x\Delta t}\widehat f^{\,n}(k,v_x,v_y).
```

For nonzero ``k``, the field increments are computed from
``(\widehat\rho^{\,n+1}-\widehat\rho^{\,n})/(ik)`` and
``(\widehat q^{\,n+1}-\widehat q^{\,n})/(ik)``, respectively, where
``\rho=\int f\,dv_x\,dv_y`` and ``q=\int(v_y/v_x)f\,dv_x\,dv_y``.
The velocity nodes must therefore have nonzero `vx`. Both electric-field
zero modes are set to zero.

Update `f`, `Ex`, and `Ey` in place. `f` has dimensions `(nvx, nvy, nx)`;
fields are vectors of length `nx`.
"""
function solve!(f, Ex, Ey, hf::OperatorHF, dt)

    nvx, nvy, nx = size(f)

    dvx = hf.mesh.dvx
    dvy = hf.mesh.dvy

    hf.rhoh .= vec(sum(f; dims = [1, 2])) .* dvx .* dvy

    for j in axes(f, 2)
        hf.ft .= f[:, j, :]
        hf.fnsvy[:, j, :] .= hf.mesh.vy[j] .* real(hf.ft) ./ hf.mesh.vx
        hf.ft .= fft(hf.ft, 2)
        hf.ft .*= exp.(-1im .* hf.mesh.vx .* hf.kx' * dt)
        hf.ft .= ifft(hf.ft, 2)
        f[:, j, :] .= real(hf.ft)
        hf.fnp1vy[:, j, :] .= hf.mesh.vy[j] .* real(hf.ft) ./ hf.mesh.vx
    end

    hf.rhon .= vec(sum(f; dims = [1, 2])) .* dvx .* dvy
    hf.rhohvy .= vec(sum(hf.fnsvy; dims = [1, 2])) .* dvx .* dvy
    hf.rhonvy .= vec(sum(hf.fnp1vy; dims = [1, 2])) .* dvx .* dvy


    hf.rhs .= (1 ./ (1im * hf.kmod)) .* fft(hf.rhon .- hf.rhoh)
    hf.Ef .= fft(Ex)
    hf.Ef .+= hf.rhs
    hf.Ef[1] = 0
    Ex .= real(ifft(hf.Ef))

    hf.rhs .= (1 ./ (1im * hf.kmod)) .* fft(hf.rhonvy .- hf.rhohvy)
    hf.Ef .= fft(Ey)
    hf.Ef .+= hf.rhs
    hf.Ef[1] = 0
    Ey .= real(ifft(hf.Ef))

end

@doc raw"""
    hf!(f, E, dt, dv, v, kx)

Advance one-velocity free streaming and its electric-current coupling:

```math
\partial_t f+v\partial_xf=0,\qquad
\partial_t E=-J,\qquad J=\int vf\,dv.
```

Apply ``e^{-ikv\Delta t}`` in spatial Fourier space and update the nonzero
field modes using
``\widehat E^{\,n+1}=\widehat E^{\,n}
+(\widehat\rho^{\,n+1}-\widehat\rho^{\,n})/(ik)``,
where ``\rho=\int f\,dv`` is approximated by a velocity sum times `dv`.
Set the field zero mode to zero.

Update `f`, of size `(length(v), length(kx))`, and `E` in place; return `E`.
"""
function hf!(
        f::Matrix{Float64},
        E::Vector{Float64},
        dt::Float64,
        dv::Float64,
        v::Vector{Float64},
        kx::Vector{Float64},
    )
    rhoh = vec(sum(f; dims = 1)) .* dv
    ft = fft(f, 2)
    ft .*= exp.(-1im * v .* kx' * dt)
    f .= real(ifft(ft, 2))
    rhon = vec(sum(f; dims = 1)) .* dv
    Ef = fft(E)
    rhs = fft(rhon .- rhoh)

    nx = length(kx)

    for i in 2:nx
        Ef[i] = Ef[i] + rhs[i] / (1im * kx[i])
    end

    Ef[1] = 0

    return E .= real(ifft(Ef))
end
