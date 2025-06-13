using FFTW
import VlasovAmpereEnergyPreserving: poisson!, updateg, semilag_d2, compute_diag
import VlasovAmpereEnergyPreserving: hf!, HE!, HB!
import VlasovAmpereEnergyPreserving: updateg!
using Plots
using DataFrames

function vm_lie_1(tmax)
    nx = 16
    nvx = 49
    nvy = 47
    dt = 0.125 / 2
    nt = floor(Int, tmax / dt)

    kinit = 1.25
    xmin, xmax = 0, 2π / kinit
    vxmin, vxmax = -0.2, 0.2
    vymin, vymax = -0.5, 0.5

    f = zeros(nvx, nvy, nx)

    x = LinRange(xmin, xmax, nx+1)[1:(end - 1)]
    dx = x[2] - x[1]
    vx = LinRange(vxmin, vxmax, nvx+1)[1:(end - 1)]
    vy = LinRange(vymin, vymax, nvy+1)[1:(end - 1)]
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]

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

    diags = [compute_diag(f, Ex, Ey, B, vx, vy, rx, ry, dx, t)]

    while abs(t - tmax) > 1e-10

        hf!(f, Ex, Ey, dt, vx, vy, kx)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        rx, ry = HE!(f, rx, ry, B, vx, vy, gx, gy, kx, dx, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        HB!(f, Ey, B, vx, vy, kx, kvx, kvy, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        iter += 1
        t += dt

        push!(diags, compute_diag(f, Ex, Ey, B, vx, vy, rx, ry, dx, t))
    end

    return vcat(DataFrame.(diags)...)
end

# Run the simulation
tmax = 10.0
@time lie1 = vm_lie_1(tmax)
plot(lie1[!, "time"], lie1[!, "ExNorm"]; yscale=:log10, label="∫|Ex| lie1")
plot!(lie1[!, "time"][2:end], lie1[!, "EyNorm"][2:end]; yscale=:log10, label="∫|Ey| lie2")
