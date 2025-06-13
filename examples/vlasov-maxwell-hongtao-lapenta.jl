using FFTW
using LinearAlgebra
using Printf
import VlasovAmpereEnergyPreserving: updateg, poisson, semilag_d2, compute_diag
using DataFrames
using Plots
import VlasovAmpereEnergyPreserving: hf_lapenta2d!, HE_lapenta2d!, HB_lapenta2d!



# Hongtao-Lapenta splitting


function lapenta_2d(tmax)
    # Computational domain parameters
    nx = 16
    nvx = 47
    nvy = 47
    dt = 0.125 / 2
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
    gx, gy = updateg(Ex, Ey, rx, ry)

    # Useful arrays
    kvx = 2π / (vxmax - vxmin) * fftfreq(nvx, nvx) |> collect
    kvy = 2π / (vymax - vymin) * fftfreq(nvy, nvy) |> collect

    time = 0.0
    iter = 0

    diags = [compute_diag(f, B, Ey, Ex, vx, vy, ry, rx, dx, time)]

    while abs(time - tmax) > 1e-10
        iter += 1
        time += dt

        @printf("Time: %.2f, Max gy: %.4e\n", time, maximum(abs.(gy)))

        hf_lapenta2d!(f, dt, vx, kx)
        HE_lapenta2d!(f, Ex, Ey, vx, vy, dt)
        HB_lapenta2d!(f, Ex, Ey, B, vx, vy, kx, kvx, kvy, dt)

        push!(diags, compute_diag(f, B, Ey, Ex, vx, vy, ry, rx, dx, time))

        rho = sum(f, dims = [1, 2]) * dvx * dvy
        Ex_poisson = poisson(vec(rho), x, kx)
        @show iter, time, norm(Ex_poisson - Ex)

    end

    return vcat(DataFrame.(diags)...)

end


tmax = 100.0
@time lapenta = lapenta_2d(tmax)
plot(lapenta[!, "time"], lapenta[!, "TotalEnergy1"])
