struct OperatorLapentaHF{T}

    mesh::Mesh1D2V{T}
    kx::Vector{T}
    ft::Array{Complex{T}, 2}

    function OperatorLapentaHF(mesh::Mesh1D2V{T}) where {T}

        kx = collect(2π / (mesh.xmax - mesh.xmin) .* fftfreq(mesh.nx, mesh.nx))

        ft = zeros(Complex{T}, (mesh.nvx, mesh.nx))

        return new{T}(mesh, kx, ft)

    end


end

@doc raw"""
    solve!(f::AbstractArray{T,3}, hf::OperatorLapentaHF, dt::Real) where {T}

Advance free streaming, leaving all fields fixed:

```math
\partial_t f+v_x\partial_x f=0,\qquad
\partial_t E_x=\partial_t E_y=\partial_t B=0.
```

For each velocity pair, apply the exact spatial Fourier multiplier
``e^{-ikv_x\Delta t}`` on the periodic mesh.
Update `f`, with dimensions `(nvx, nvy, nx)`, in place and return `nothing`.
"""
function solve!(f::AbstractArray{T, 3}, hf::OperatorLapentaHF, dt::Real) where {T}

    nvx, nvy, nx = size(f)

    for j in axes(f, 2)

        hf.ft .= f[:, j, :]
        hf.ft .= fft(hf.ft, 2)
        hf.ft .*= exp.(-1im .* hf.mesh.vx .* hf.kx' * dt)
        hf.ft .= ifft(hf.ft, 2)

        f[:, j, :] .= real(hf.ft)
    end

    return
end
