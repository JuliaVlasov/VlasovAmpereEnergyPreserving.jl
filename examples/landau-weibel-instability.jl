include("weibel-instability.jl")

"""
Run the Landau-Weibel test from Section 6.3 of the paper.

The paper does not specify finite velocity bounds for this test. We truncate
the periodic velocity grids to [-6, 6) in vx and [-8, 8) in vy. The finite
vx grid causes recurrence near the end of the simulation.
"""
function run_landau_weibel(; tmax = 50.0, dt = 0.125)
    return run_weibel(;
        tmax, dt,
        nx = 47, nvx = 47, nvy = 47,
        k = 0.5, α = 1.0e-3, β = -1.0e-6,
        σ1 = 1.0, σ2 = sqrt(2.0),
        vxmin = -6.0, vxmax = 6.0, vymin = -8.0, vymax = 8.0,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_landau_weibel()
    drift = (result.total_energy[end] - result.total_energy[1]) / result.total_energy[1]
    println("Landau-Weibel instability completed at t = $(result.time[end])")
    println("Relative total-energy drift: $drift")
    println("Final Gauss-law residual: $(result.gauss_error[end])")
end
