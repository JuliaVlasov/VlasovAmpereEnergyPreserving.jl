using .Threads

struct OperatorLapentaHB{T}

    mesh::Mesh1D2V{T}
    kx::Vector{T}
    kvx::Vector{T}
    kvy::Vector{T}
    ft::Array{Complex{T}, 3}

    function OperatorLapentaHB(mesh::Mesh1D2V{T}) where T

        kx = collect(2π / (mesh.xmax - mesh.xmin) * fftfreq(mesh.nx, mesh.nx))
        kvx = collect(2π / (mesh.vxmax - mesh.vxmin) * fftfreq(mesh.nvx, mesh.nvx))
        kvy = collect(2π / (mesh.vymax - mesh.vymin) * fftfreq(mesh.nvy, mesh.nvy))

        ft = zeros(Complex{T}, (mesh.nvx, mesh.nvy, mesh.nx))

        new{T}(mesh, kx, kvx, kvy, ft)

    end

end


"""
solve the subsystem 

```math
    df/dt+B Jv.df/dv=0 
    dEy/dt = -dB/dx
    dB/dt  = -dEy/dx 
    dEx/dt = 0
```

first equation is solved with semi-Lagrangian + exact splitting

"""
function solve!(f, Ex, Ey, B, hb::OperatorLapentaHB, dt)

    hdt = 0.5dt

    Eyf = fft(Ey)
    Bf = fft(B)
    Eynp1f = similar(Eyf)
    Bnp1f = similar(Bf)

    for i in eachindex(hb.kx)
        c = 0.5dt * 1im * hb.kx[i]

        AA = [1 c; c 1]
        Un = [Eyf[i] - c * Bf[i], Bf[i] - c * Eyf[i]]

        U = AA \ Un
        Eynp1f[i] = U[1]
        Bnp1f[i] = U[2]
    end

    Eynp1 = real(ifft(Eynp1f))
    Bnp1 = real(ifft(Bnp1f))

    Bdemi = 0.5 * (Bnp1 .+ B)

    hb.ft .= fft(f, 1)
    for l in axes(hb.ft, 3), j in axes(hb.ft, 2), i in axes(hb.ft, 1)
        hb.ft[i, j, l] = exp(-1im * hb.kvx[i] * hb.mesh.vy[j] * tan(Bdemi[l] * hdt)) * hb.ft[i, j, l]
    end
    hb.ft .= ifft(hb.ft, 1)

    hb.ft .= fft(hb.ft, 2)
    for l in axes(hb.ft, 3), j in axes(hb.ft, 2), i in axes(hb.ft, 1)
        hb.ft[i, j, l] = exp(1im * hb.kvy[j] * hb.mesh.vx[i] * sin(Bdemi[l] * dt)) * hb.ft[i, j, l]
    end
    hb.ft .= ifft(hb.ft, 2)

    hb.ft .= fft(hb.ft, 1)
    for l in axes(hb.ft, 3), j in axes(hb.ft, 2), i in axes(hb.ft, 1)
        hb.ft[i, j, l] = exp(- 1im * hb.kvx[i] * hb.mesh.vy[j] * tan(Bdemi[l] * hdt)) * hb.ft[i, j, l]
    end
    hb.ft .= ifft(hb.ft, 1)

    Ey .= Eynp1
    B .= Bnp1
    f .= real(hb.ft)

    return 
end

