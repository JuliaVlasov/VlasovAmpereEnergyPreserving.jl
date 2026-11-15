using FFTW
import VlasovAmpereEnergyPreserving: Mesh1D2V, OperatorHF, OperatorHE, solve!, poisson!, updateg!

const WEIBEL_D = 9

# Weights depend only on the displacement, not on the velocity-grid index.
function weibel_semilag(f::Vector{Float64}, velocity::Float64, dt::Float64, dv::Float64)
    shift = -velocity * dt / dv
    offset = floor(Int, shift)
    α = shift - offset
    stencil = -WEIBEL_D:(WEIBEL_D + 1)
    weights = [
        prod((α - j) / (i - j) for j in stencil if j != i)
        for i in stencil
    ]
    n = length(f)
    result = similar(f)
    for i in eachindex(result)
        value = 0.0
        for (j, weight) in zip(stencil, weights)
            value += weight * f[mod1(i + offset + j, n)]
        end
        result[i] = value
    end
    return result
end

# Magnetic velocity rotation as the three semi-Lagrangian shears in Section 3.2.3.
function weibel_magnetic!(f, Ey, B, mesh, kx, dt)
    for l in eachindex(mesh.x)
        for j in eachindex(mesh.vy)
            speed = mesh.vy[j] * tan(B[l] * dt / 2)
            f[:, j, l] .= weibel_semilag(f[:, j, l], speed, 1.0, mesh.dvx)
        end
        for i in eachindex(mesh.vx)
            speed = -mesh.vx[i] * sin(B[l] * dt)
            f[i, :, l] .= weibel_semilag(f[i, :, l], speed, 1.0, mesh.dvy)
        end
        for j in eachindex(mesh.vy)
            speed = mesh.vy[j] * tan(B[l] * dt / 2)
            f[:, j, l] .= weibel_semilag(f[:, j, l], speed, 1.0, mesh.dvx)
        end
    end
    Ey .-= dt .* real(ifft(1im .* kx .* fft(B)))
end

"""
Run the Vlasov-Maxwell Weibel test from Section 6.2 of the paper.

The periodic velocity grids omit their upper endpoints. As in the existing
Vlasov-Maxwell example, a constant of 1 is added to each SAV electric energy
to avoid division by zero for the initially vanishing electric fields.
The even spatial grid cannot represent the derivative of its real Nyquist
mode, which contributes to the measured Gauss-law residual at later times.
"""
function run_weibel(;
        tmax = 300.0, dt = 1 / 16,
        nx = 16, nvx = 47, nvy = 47,
        k = 1.25, α = 0.0, β = -1.0e-4,
        σ1 = 0.02 / sqrt(2), σ2 = sqrt(12) * σ1,
        vxmin = -0.2, vxmax = 0.2, vymin = -0.5, vymax = 0.5,
    )
    nsteps = round(Int, tmax / dt)
    isapprox(nsteps * dt, tmax) ||
        throw(ArgumentError("tmax must be an integer multiple of dt"))

    mesh = Mesh1D2V(0.0, 2π / k, nx, vxmin, vxmax, nvx, vymin, vymax, nvy)
    kx = collect(2π / (mesh.xmax - mesh.xmin) .* fftfreq(nx, nx))

    f = zeros(nvx, nvy, nx)
    for i in eachindex(mesh.vx), j in eachindex(mesh.vy), l in eachindex(mesh.x)
        f[i, j, l] =
            exp(-0.5 * (mesh.vx[i]^2 / σ1^2 + mesh.vy[j]^2 / σ2^2)) *
            (1 + α * cos(k * mesh.x[l])) / (2π * σ1 * σ2)
    end

    rho = vec(sum(f; dims = (1, 2))) .* (mesh.dvx * mesh.dvy) .- 1
    Ex = zeros(nx)
    poisson!(Ex, rho, kx)
    Ey = zeros(nx)
    B = β .* cos.(k .* mesh.x)

    rx = sqrt(1 + 0.5 * sum(Ex .^ 2) * mesh.dx)
    ry = sqrt(1 + 0.5 * sum(Ey .^ 2) * mesh.dx)
    gx = similar(Ex)
    gy = similar(Ey)
    hf = OperatorHF(mesh)
    he = OperatorHE(mesh)

    time = collect(0:nsteps) .* dt
    magnetic_norm = zeros(nsteps + 1)
    ex_norm = zeros(nsteps + 1)
    ey_norm = zeros(nsteps + 1)
    total_energy = zeros(nsteps + 1)
    gauss_error = zeros(nsteps + 1)

    function record!(i)
        magnetic_norm[i] = sum(B .^ 2) * mesh.dx
        ex_norm[i] = sum(Ex .^ 2) * mesh.dx
        ey_norm[i] = sum(Ey .^ 2) * mesh.dx
        kinetic = sum(
            (mesh.vx[a]^2 + mesh.vy[b]^2) * f[a, b, l]
            for a in eachindex(mesh.vx), b in eachindex(mesh.vy), l in eachindex(mesh.x)
        ) * (0.5 * mesh.dx * mesh.dvx * mesh.dvy)
        total_energy[i] = kinetic + rx^2 + ry^2 + 0.5 * magnetic_norm[i] - 2
        rho = vec(sum(f; dims = (1, 2))) .* (mesh.dvx * mesh.dvy) .- 1
        gauss_error[i] = sqrt(sum((real(ifft(1im .* kx .* fft(Ex))) .- rho) .^ 2) * mesh.dx)
    end
    record!(1)

    for step in 1:nsteps
        solve!(f, Ex, Ey, hf, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)
        rx, ry = solve!(f, rx, ry, B, gx, gy, he, dt; interpolate = weibel_semilag)
        weibel_magnetic!(f, Ey, B, mesh, kx, dt)
        record!(step + 1)
    end

    return (; time, magnetic_norm, ex_norm, ey_norm, total_energy, gauss_error, f, Ex, Ey, B)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_weibel()
    drift = (result.total_energy[end] - result.total_energy[1]) / result.total_energy[1]
    println("Weibel instability completed at t = $(result.time[end])")
    println("Relative total-energy drift: $drift")
    println("Final Gauss-law residual: $(result.gauss_error[end])")
end
