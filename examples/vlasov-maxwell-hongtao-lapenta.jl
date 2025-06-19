using FFTW
using LinearAlgebra
using Printf
import VlasovAmpereEnergyPreserving: updateg, poisson!, semilag_d2, compute_diag
using DataFrames
using Plots
import VlasovAmpereEnergyPreserving: hf_lapenta2d!, HE_lapenta2d!, HB_lapenta2d!
using VlasovAmpereEnergyPreserving

# Hongtao-Lapenta splitting

function lapenta_2d(tmax)
    # Computational domain parameters
    nx = 16
    nvx = 49
    nvy = 47
    dt = 0.125 / 2
    nt = floor(Int, tmax / dt)

    # Test case selection
    kinit = 1.25
    xmin, xmax = 0.0, 2π / kinit
    vxmin, vxmax = -0.2, 0.2
    vymin, vymax = -0.5, 0.5

    mesh = Mesh1D2V(xmin, xmax, nx, vxmin, vxmax, nvx, vymin, vymax, nvy)

    x = mesh.x
    vx = mesh.vx
    vy = mesh.vy
    dx = mesh.dx
    dvx = mesh.dvx
    dvy = mesh.dvy

    kx = collect(2π / (xmax - xmin) * fftfreq(nx, nx))

    f = zeros(nvx, nvy, nx)

    α = 0.0001
    σ1 = 0.02 / sqrt(2)
    σ2 = sqrt(12) * σ1
    for i in eachindex(vx), j in eachindex(vy), k in eachindex(x)
        f[i, j, k] =
            exp(-0.5 * vx[i]^2 / σ1^2) * exp(-0.5 * vy[j]^2 / σ2^2) * (1 + α * cos(kinit * x[k])) /
            (2π * σ1 * σ2)
    end

    Ex = zeros(nx)
    Ey = zeros(nx)
    B = zeros(nx)
    rho = vec(sum(f; dims = (1, 2))) * dvx * dvy
    poisson!(Ex, rho, kx)
    B = α * cos.(kinit .* x)

    rx = sqrt(0.5 * sum(Ex .^ 2) * dx + 1.0)
    ry = sqrt(0.5 * sum(Ey .^ 2) * dx + 1.0)

    # Useful arrays
    kvx = collect(2π / (vxmax - vxmin) * fftfreq(nvx, nvx))
    kvy = collect(2π / (vymax - vymin) * fftfreq(nvy, nvy))

    time = 0.0
    iter = 0

    diags = [compute_diag(f, Ex, Ey, B, rx, ry, mesh, time)]

    while abs(time - tmax) > 1.0e-10
        hf_lapenta2d!(f, kx, mesh, dt)
        HE_lapenta2d!(f, Ex, Ey, mesh, dt)
        HB_lapenta2d!(f, Ex, Ey, B, kx, kvx, kvy, mesh, dt)

        iter += 1
        time += dt

        push!(diags, compute_diag(f, Ex, Ey, B, rx, ry, mesh, time))
    end

    return vcat(DataFrame.(diags)...)
end
