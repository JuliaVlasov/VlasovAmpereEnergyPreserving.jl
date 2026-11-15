using Test
using CUDA
include("solver.jl")
include("../../examples/weibel-instability.jl")

@testset "Paper-style interpolation" begin
    data = reshape(Float64[1, 2, 4, 8], 4, 1, 1)
    @test vec(advect_linear(data, fill(0.5, 1, 1, 1), 1, 1.0)) == [1, 1.5, 3, 6]
    @test vec(advect_linear(data, fill(-0.5, 1, 1, 1), 1, 1.0)) == [1.5, 3, 6, 8]

    vline = collect(-3.0:3.0)
    profile = Float64[0, 0, 1, 2, 1, 0, 0]
    shift = 0.3
    moved = vec(advect_linear(reshape(profile, 7, 1, 1), fill(shift, 1, 1, 1), 1, 1.0))
    mass = sum(profile)
    initial_momentum = sum(vline .* profile)
    kinetic_change = 0.5 * sum(vline .^ 2 .* (moved .- profile))
    exact_translation = shift * initial_momentum + 0.5 * shift^2 * mass
    @test sum(moved) ≈ mass
    @test kinetic_change - exact_translation ≈ 0.5 * shift * (1 - shift) * mass

    n = 23
    dv = 2.0 / n
    v = collect(range(-1.0; step = dv, length = n))
    f = reshape([sin(2π * (i - 1) / n) * cos(4π * (j - 1) / n)
                 for i in 1:n, j in 1:n, l in 1:1], n, n, 1)
    filter = 3 ./ (2 .+ cos.(2π .* fftfreq(n, n) ./ n))
    fx, fy = reshape(filter, n, 1, 1), reshape(filter, 1, n, 1)
    unchanged = rotate_cubic(f, [0.0], v, v, -1.0, -1.0, dv, dv, 1.0, fx, fy)
    @test unchanged ≈ f atol = 1.0e-12
    rotated = rotate_cubic(f, [π / 2], v, v, -1.0, -1.0, dv, dv, 1.0, fx, fy)
    expected = [f[mod1(2 - j, n), i, 1] for i in 1:n, j in 1:n]
    @test rotated[:, :, 1] ≈ expected atol = 1.0e-12

    nx = 16
    nyquist = (-1.0) .^ (0:(nx - 1))
    derivative = real.(ifft(im .* fftfreq(nx, nx) .* fft(nyquist)))
    @test maximum(abs, derivative) < 1.0e-12
end

@testset "CUDA benchmark numerical backend" begin
    for (case, nx, dt, kwargs) in (
            ("weibel", 16, 1 / 16, (;)),
            ("landau-weibel", 47, 0.125,
                (; k = 0.5, α = 1.0e-3, β = -1.0e-6, σ1 = 1.0, σ2 = sqrt(2.0),
                    vxmin = -6.0, vxmax = 6.0, vymin = -8.0, vymax = 8.0)),
        )
        reference = run_weibel(; nx, nvx = 47, nvy = 47, dt, tmax = 1.0, kwargs...)
        cpu = simulate_vm(case; nx, nv = 47, dt, tmax = 1.0, T = Float64,
                          diagnostics = true, method = :high_order)
        @test cpu.f ≈ reference.f rtol = 1.0e-11 atol = 1.0e-11
        @test cpu.Ex ≈ reference.Ex atol = 1.0e-12
        @test cpu.Ey ≈ reference.Ey atol = 1.0e-12
        @test cpu.B ≈ reference.B atol = 1.0e-12
        @test cpu.history[end].total_energy ≈ reference.total_energy[end] atol = 1.0e-12

        single = simulate_vm(case; nx, nv = 47, dt, tmax = 1.0, diagnostics = true,
                             method = :high_order)
        @test sqrt(sum(abs2, single.f .- cpu.f) / sum(abs2, cpu.f)) < 1.0e-4
        matched = simulate_vm(case; nx, nv = 47, dt, tmax = 1.0, T = Float64, diagnostics = true)
        @test sqrt(sum(abs2, matched.f .- reference.f) / sum(abs2, reference.f)) < 2.0e-4
        @test matched.history[end].magnetic_norm ≈ reference.magnetic_norm[end] rtol = 1.0e-4
        if CUDA.functional()
            CUDA.allowscalar(false)
            gpu = simulate_vm(case; nx, nv = 47, dt, tmax = 1.0, array_type = CuArray,
                              diagnostics = true, method = :high_order)
            @test Array(gpu.f) ≈ single.f rtol = 1.0e-3 atol = 1.0e-5
            @test Array(gpu.B) ≈ single.B rtol = 0.1 atol = 1.0e-7
            matched_cpu = simulate_vm(case; nx, nv = 47, dt, tmax = 1.0)
            matched_gpu = simulate_vm(case; nx, nv = 47, dt, tmax = 1.0, array_type = CuArray)
            @test Array(matched_gpu.f) ≈ matched_cpu.f rtol = 1.0e-3 atol = 1.0e-5
        else
            @test_skip CUDA.functional()
        end
    end
end
