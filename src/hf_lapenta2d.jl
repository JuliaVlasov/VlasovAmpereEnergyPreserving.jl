struct OperatorLapentaHF{T}

    mesh::Mesh1D2V{T}
    kx::Vector{T}
    ft::Array{Complex{T}, 2}

    function OperatorLapentaHF(mesh::Mesh1D2V{T}) where T

        kx = collect(2π / (mesh.xmax - mesh.xmin) .* fftfreq(mesh.nx, mesh.nx))

        ft = zeros(Complex{T}, (mesh.nvx, mesh.nx))

        new{T}( mesh, kx, ft)

    end


end

"""
solve the subsystem 

```math
df/dt + vx df/dx = 0 
dEx/dt = 0
dEy/dt = 0
dB/dt  = 0
```
"""
function solve!( f::AbstractArray{T, 3}, hf::OperatorLapentaHF, dt::Real) where {T}

    nvx, nvy, nx = size(f)

    for j in axes(f, 2)

        hf.ft .= f[:, j, :]
        hf.ft .= fft(hf.ft, 2)
        hf.ft .*= exp.(-1im .* hf.mesh.vx .* hf.kx' * dt)
        hf.ft .= ifft(hf.ft, 2)

        f[:, j, :] .= real(hf.ft)
    end

end

