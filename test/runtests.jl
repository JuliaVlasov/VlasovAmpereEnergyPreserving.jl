using VlasovAmpereEnergyPreserving
using Test
import VlasovAmpereEnergyPreserving: semilag_d9

@testset "VlasovAmepereEnergyPreserving.jl" begin
    # Write your tests here.
end

include("../examples/landau-weibel-instability.jl")
include("../examples/two-stream-instability.jl")

@testset "Two-stream example" begin
    initial = run_two_stream(; tmax = 0.0)
    v = -6.0 + 74 * 12.0 / 128
    expected = v^2 * exp(-0.5 * v^2) * 1.05 / sqrt(2π)
    @test initial.f[75, 1] ≈ expected
    @test initial.f[65, 1] == 0.0
    result = run_two_stream(; tmax = 20.0)
    @test all(isfinite, result.f)
    @test result.electric_energy[end] > result.electric_energy[1]
end

@testset "Weibel example" begin
    f = sin.(2π .* (0:46) ./ 47)
    for speed in (-2.4, -0.02, 0.0, 0.03, 1.7)
        @test weibel_semilag(f, speed, 0.125, 0.2) ≈ semilag_d9(f, speed, 0.125, 0.2) atol = 1.0e-12
    end

    result = run_weibel(; tmax = 1.0)
    @test result.time[end] == 1.0
    @test all(isfinite, result.f)
    @test result.gauss_error[end] < 1.0e-10
    @test abs(result.total_energy[end] - result.total_energy[1]) < 1.0e-10
    @test result.magnetic_norm[end] != result.magnetic_norm[1]
end

@testset "Landau-Weibel example" begin
    initial = run_landau_weibel(; tmax = 0.0)
    @test size(initial.f) == (47, 47, 47)
    @test maximum(abs, initial.B) ≈ 1.0e-6
    @test initial.ex_norm[1] > 0
    @test initial.ey_norm[1] == 0
    vx = -6.0 + 23 * 12.0 / 47
    vy = -8.0 + 23 * 16.0 / 47
    expected = exp(-0.5 * (vx^2 + vy^2 / 2)) * (1 + 1.0e-3) / (2π * sqrt(2))
    @test initial.f[24, 24, 1] ≈ expected

    result = run_landau_weibel(; tmax = 16.0)
    @test result.time[end] == 16.0
    @test all(isfinite, result.f)
    @test result.ex_norm[end] < result.ex_norm[1]
    @test result.ey_norm[end] > 0
    damping_rate = log(result.ex_norm[110] / result.ex_norm[39]) / (2 * (13.625 - 4.75))
    @test damping_rate ≈ -0.153 atol = 0.01
    @test result.magnetic_norm[end] > result.magnetic_norm[1]
end
