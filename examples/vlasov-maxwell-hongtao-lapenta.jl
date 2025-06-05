using FFTW
using LinearAlgebra
using Printf
import VlasovAmpereEnergyPreserving: updateg, poisson, semilag_d2, compute_diag
using DataFrames

# Hongtao-Lapenta splitting


function hf_lapenta2d!(f, dt, vx, kx)

    for j in axes(f, 2)

        fs = fft(f[:, j, :], 1)
        fs .*= exp.(-1im * vx .* kx' * dt)
        fs .= ifft(fs, 1)

        f[:, j, :] .= real(fs)
    end

end


"""
solve the subsystem 

math```
    df/dt + Ex df/dvx + Ey df/dvy = 0
    dEx/dt=-Jx
    dEy/dt=-Jy
    dB/dt=0
```

"""
function HE_lapenta2d!(f, Ex, Ey, vx, vy, dt)

    nx = length(Ex)
    nvx = length(vx)
    nvy = length(vy)
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]

    Jx = zeros(nx)
    Jy = zeros(nx)
    rho = zeros(nx)
    for i in axes(f, 3), j in axes(f, 2), k in axes(f, 1)
        Jx[i] += vx[j] * f[j, k, i]
        Jy[i] += vy[k] * f[j, k, i]
        rho[i] += f[j, k, i]
    end

    Jx .*= dvx * dvy
    Jy .*= dvx * dvy
    rho .*= dvx * dvy

    dt2_rho4 = (dt^2) .* rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2

    Exnew = (1 ./ (1 .+ dt2_rho4)) .* Ex .- (dt ./ (2 .+ dt2_rho2)) .* Jx
    Eynew = (1 ./ (1 .+ dt2_rho4)) .* Ey .- (dt ./ (2 .+ dt2_rho2)) .* Jy

    for j in axes(f, 2), i in eachindex(Exnew)
        ft = f[:, j, i]
        u1 = semilag_d2(ft, nvx, Exnew[i], dt, dvx)
        f[:, j, i] .= u1
    end

    for k in axes(f, 1), i in eachindex(Eynew)
        ft = f[k, :, i]
        u1 = semilag_d2(ft, nvy, Eynew[i], dt, dvy)
        f[k, :, i] .= u1
    end

    Ex .= 2 .* Exnew .- Ex
    Ey .= 2 .* Eynew .- Ey


end


"""
solve the subsystem 

    df/dt+B Jv.df/dv=0 (with semi-Lagrangian + exact splitting) 
    dEy/dt = -dB/dx
    dB/dt  = -dEy/dx 
    dEx/dt = 0

"""
function HB_lapenta2d!(f, Ex, Ey, B, vx, vy, kx, kvx, kvy, dt)

    Eyf = fft(Ey)
    Bf = fft(B)
    Eynp1f = similar(Eyf)
    Bnp1f = similar(Bf)

    for i in eachindex(kx)
        AA=[
            1 0.5dt * 1im * kx[i];
            0.5dt * 1im * kx[i] 1
        ]

        Un = [
            Eyf[i] - 0.5dt * 1im * kx[i] * Bf[i]
            Bf[i] - 0.5dt * 1im * kx[i] * Eyf[i]
        ]
        U = AA \ Un
        Eynp1f[i]=U[1]
        Bnp1f[i]=U[2]
    end

    Eynp1=real(ifft(Eynp1f))
    Bnp1=real(ifft(Bnp1f))

    Bdemi = 0.5 * (Bnp1 + B)

    fn = fft(f, 1)
    for l in axes(fn, 3), j in axes(fn, 2), i in axes(fn, 1)
        fn[i, j, l] = exp(-1im * kvx[i] * vy[j] * tan(Bdemi[l]*dt/2)) * fn[i, j, l]
    end
    fn = ifft(fn, 1)

    fn = fft(fn, 2)
    for l in axes(fn, 3), j in axes(fn, 2), i in axes(fn, 1)
        fn[i, j, l] = exp(1im * kvy[j] * vx[i] * sin(Bdemi[l]*dt)) * fn[i, j, l]
    end
    fn = ifft(fn, 2)

    fn = fft(fn, 1)
    for l in axes(fn, 3), j in axes(fn, 2), i in axes(fn, 1)
        fn[i, j, l] = exp(- 1im * kvx[i] * vy[j] * tan(Bdemi[l]*dt/2)) * fn[i, j, l]
    end
    fn = ifft(fn, 1)

    Ey .= Eynp1
    B .= Bnp1
    f .= real(fn)

end

function main()
    # Computational domain parameters
    nx = 16
    nvx = 47
    nvy = 47
    dt = 0.125 / 2
    tmax = 300.0
    nt = floor(Int, tmax / dt)

    # Test case selection
    kinit = 1.25
    vxmin, vxmax = -0.2, 0.2
    vymin, vymax = -0.5, 0.5

    xmin, xmax = 0, 2π / kinit
    dx = (xmax-xmin) / nx
    x = LinRange(xmin, xmax, nx+1)[1:(end-1)]

    kx = 2π/(xmax-xmin) * fftfreq(nx, nx) |> collect

    dvx = (vxmax - vxmin) / nvx
    dvy = (vymax - vymin) / nvy
    vx = LinRange(vxmin, vxmax, nvx+1)[1:(end-1)]
    vy = LinRange(vymin, vymax, nvy+1)[1:(end-1)]

    f = zeros(nvx, nvy, nx)

    α = 0.0001
    σ1 = 0.02/sqrt(2)
    σ2 = sqrt(12)*σ1
    for i in eachindex(vx), j in eachindex(vy), k in eachindex(x)
        f[i, j, k] =
            exp(-0.5*vx[i]^2/σ1^2) * exp(-0.5*vy[j]^2/σ2^2) * (1 + α*cos(kinit*x[k])) /
            (2π*σ1*σ2)
    end

    Ex = zeros(nx)
    Ey = zeros(nx)
    B = zeros(nx)
    rho = vec(sum(f, dims = (1, 2))) * dvx * dvy
    Ex = poisson(rho, x, kx)
    B = α * cos.(kinit .* x)

    rx = sqrt(0.5 * sum(Ex .^ 2) * dx + 1.0)
    ry = sqrt(0.5 * sum(Ey .^ 2) * dx + 1.0)
    gx, gy = updateg(rx, Ex, ry, Ey, x, nx)

    # Useful arrays
    kvx = 2π / (vxmax - vxmin) * fftfreq(nvx, nvx) |> collect
    kvy = 2π / (vymax - vymin) * fftfreq(nvy, nvy) |> collect

    time = 0.0
    iter = 0

    diags = [compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, nx, nvx, nvy, dx, time)]

    while abs(time - tmax) > 1e-10
        iter += 1
        time += dt

        @printf("Time: %.2f, Max gy: %.4e\n", time, maximum(abs.(gy)))

        hf_lapenta2d!(f, dt, vx, kx)
        HE_lapenta2d!(f, Ex, Ey, vx, vy, dt)
        HB_lapenta2d!(f, Ex, Ey, B, vx, vy, kx, kvx, kvy, dt)

        push!(diags, compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, nx, nvx, nvy, dx, time))

        rho = sum(f, dims = [1, 2]) * dvx * dvy
        Ex_poisson = poisson(vec(rho), x, kx)
        @show iter, time, norm(Ex_poisson - Ex)

    end

    return vcat(DataFrame.(diags)...)

end


@time results = main()
