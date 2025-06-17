
using FFTW
using LinearAlgebra
using Printf
using BenchmarkTools
using .Threads
import VlasovAmpereEnergyPreserving: poisson!, updateg, semilag_d2, compute_diag
import VlasovAmpereEnergyPreserving: HE2!, HB2!, updateg!, Mesh1D2V, solve!, poisson!
import VlasovAmpereEnergyPreserving: OperatorHF



function vm_lie_2(tmax)
    nx = 16
    nvx = 47
    nvy = 47
    dt = 0.125 / 2
    nt = floor(Int, tmax / dt)

    kinit = 1.25
    xmin, xmax = 0.0, 2π / kinit
    vxmin, vxmax = -0.2, 0.2
    vymin, vymax = -0.5, 0.5

    f = zeros(nvx, nvy, nx)

    mesh = Mesh1D2V(xmin, xmax, nx, vxmin, vxmax, nvx, vymin, vymax, nvy)

    x = mesh.x
    kx = 2π / (xmax - xmin) * collect(fftfreq(nx, nx))
    vx = mesh.vx
    vy = mesh.vy

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
    dvx = mesh.dvx
    dvy = mesh.dvy
    rho = vec(sum(f; dims=(1, 2))) * dvx * dvy
    poisson!(Ex, rho, kx)
    B = α * cos.(kinit .* x)

    kvx = collect(2π / (vxmax - vxmin) * fftfreq(nvx, nvx))
    kvy = collect(2π / (vymax - vymin) * fftfreq(nvy, nvy))

    dx = mesh.dx
    rx=sqrt(0.5*sum(Ex .* Ex)*dx+1.0)
    ry=sqrt(0.5*sum(Ey .* Ey)*dx+1.0)

    gx, gy = updateg(Ex, Ey, rx, ry)

    t = 0.0
    iter = 0

    diags = [compute_diag(f, Ex, Ey, B, rx, ry, mesh, t)]

    hf = OperatorHF(mesh)

    while abs(t - tmax) > 1e-10
        iter += 1
        t += dt

        solve!(f, Ex, Ey, hf, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        rx, ry = HE2!(f, Ex, Ey, gx, gy, rx, ry, mesh, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        rx, ry = HB2!(f, Ey, B, gx, gy, rx, ry, kx, kvx, kvy, mesh, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        push!(diags, compute_diag(f, Ex, Ey, B, rx, ry, mesh, t))
    end

    return vcat(DataFrame.(diags)...)
end

