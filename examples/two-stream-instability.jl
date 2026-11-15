using FFTW
import VlasovAmpereEnergyPreserving: poisson!, hf!, HE!, semilag_d9

"""
Run the Vlasov-Ampère two-stream-instability test from Section 6.1.

The paper specifies an unbounded velocity domain; `vmin` and `vmax` provide a
finite computational truncation. The default `[-6, 6)` captures the Maxwellian
to its negligible tails. The velocity transport uses the paper's SL-(2d+1)
interpolation with d = 9.
"""
function run_two_stream(;
        nx = 128,
        nv = 128,
        tmax = 100.0,
        dt = 0.5,
        k = 0.5,
        α = 0.05,
        vmin = -6.0,
        vmax = 6.0,
    )
    nsteps = round(Int, tmax / dt)
    isapprox(nsteps * dt, tmax) ||
        throw(ArgumentError("tmax must be an integer multiple of dt"))

    xmin = 0.0
    xmax = 2π / k
    dx = (xmax - xmin) / nx
    x = collect(range(xmin; step = dx, length = nx))
    kx = collect(2π / (xmax - xmin) .* fftfreq(nx, nx))

    dv = (vmax - vmin) / nv
    v = collect(range(vmin; step = dv, length = nv))
    kv = collect(2π / (vmax - vmin) .* fftfreq(nv, nv))

    feq = v .^ 2 .* exp.(-0.5 .* v .^ 2) ./ √(2π)
    f = feq .* (1 .+ α .* cos.(k .* x'))

    rho = vec(sum(f; dims = 1)) .* dv .- 1
    E = zeros(nx)
    poisson!(E, rho, kx)
    r = √(0.5 * sum(E .^ 2) * dx)

    time = collect(0:nsteps) .* dt
    electric_energy = zeros(nsteps + 1)
    total_energy = zeros(nsteps + 1)
    poisson_error = zeros(nsteps + 1)

    function record!(i)
        rho = vec(sum(f; dims = 1)) .* dv .- 1
        dEdx = real(ifft(1im .* kx .* fft(E)))
        electric_energy[i] = 0.5 * sum(E .^ 2) * dx
        kinetic_energy = 0.5 * sum(v .^ 2 .* f) * dv * dx
        total_energy[i] = kinetic_energy + r^2
        poisson_error[i] = √(sum((dEdx .- rho) .^ 2) * dx)
    end
    record!(1)

    for step in 1:nsteps
        hf!(f, E, dt / 2, dv, v, kx)
        r = HE!(f, r, v, E ./ r, nv, kv, dv, nx, dx, dt; interpolate = semilag_d9)
        hf!(f, E, dt / 2, dv, v, kx)
        record!(step + 1)
    end

    return (; time, electric_energy, total_energy, poisson_error, f, E)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_two_stream()
    relative_energy_drift =
        (result.total_energy[end] - result.total_energy[1]) / result.total_energy[1]
    println("Two-stream instability completed at t = $(result.time[end])")
    println("Relative total-energy drift: $relative_energy_drift")
    println("Final Poisson residual: $(result.poisson_error[end])")
end
