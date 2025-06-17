using FFTW
import VlasovAmpereEnergyPreserving: poisson!, updateg, semilag_d2, compute_diag
import VlasovAmpereEnergyPreserving: hf!, HE!, HB!
import VlasovAmpereEnergyPreserving: updateg!, Mesh1D2V
import VlasovAmpereEnergyPreserving: OperatorHF, solve!
using Plots
using DataFrames

function vm_lie_1(tmax)
    nx = 16
    nvx = 49
    nvy = 47
    dt = 0.125 / 2
    nt = floor(Int, tmax / dt)

    kinit = 1.25
    xmin, xmax = 0.0, 2π / kinit
    vxmin, vxmax = -0.2, 0.2
    vymin, vymax = -0.5, 0.5

    mesh = Mesh1D2V(xmin, xmax, nx, vxmin, vxmax, nvx, vymin, vymax, nvy)

    f = zeros(nvx, nvy, nx)

    x = mesh.x
    dx = mesh.dx
    vx = mesh.vx
    vy = mesh.vy
    dvx = mesh.dvx
    dvy = mesh.dvy

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

    kx = collect(2π / (xmax - xmin) * fftfreq(nx, nx))
    kvx = collect(2π / (vxmax - vxmin) * fftfreq(nvx, nvx))
    kvy = collect(2π / (vymax - vymin) * fftfreq(nvy, nvy))

    rho = vec(sum(f; dims=(1, 2))) * dvx * dvy
    poisson!(Ex, rho, kx)
    B = α * cos.(kinit .* x)

    rx = sqrt(0.5*sum(Ex .* Ex)*dx+1.0)
    ry = sqrt(0.5*sum(Ey .* Ey)*dx+1.0)

    gx, gy = updateg(Ex, Ey, rx, ry)

    t = 0.0
    iter = 0

    diags = [compute_diag(f, Ex, Ey, B, rx, ry, mesh, t)]

    hf = OperatorHF(mesh)

    while abs(t - tmax) > 1e-10

        solve!(f, Ex, Ey, hf, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        rx, ry = HE!(f, rx, ry, B, gx, gy, kx, mesh, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        HB!(f, Ey, B, mesh, kx, kvx, kvy, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        iter += 1
        t += dt

        push!(diags, compute_diag(f, Ex, Ey, B, rx, ry, mesh, t))
    end

    return vcat(DataFrame.(diags)...)
end

