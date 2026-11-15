using CUDA
include("solver.jl")

function main(args)
    length(args) in (4, 5, 6) ||
        error("Usage: julia --project=benchmarks/cuda benchmarks/cuda/run.jl CASE NX NV DT [RUNS [paper|high_order]]")
    case = args[1]
    nx, nv = parse(Int, args[2]), parse(Int, args[3])
    dt = parse(Float64, args[4])
    runs = length(args) >= 5 ? parse(Int, args[5]) : 100
    method = length(args) == 6 ? Symbol(args[6]) : :paper
    method in (:paper, :high_order) || throw(ArgumentError("method must be paper or high_order"))
    nx > 0 && nv > 0 && dt > 0 && runs > 0 ||
        throw(ArgumentError("NX, NV, DT and RUNS must be positive"))
    CUDA.functional() || error("A functional NVIDIA GPU and CUDA driver are required")
    CUDA.allowscalar(false)
    benchmark_parameters(case)

    # Compile and synchronize before the timed runs.
    simulate_vm(case; nx, nv, dt, tmax = dt, array_type = CuArray, method)
    times = Float64[]
    for _ in 1:runs
        push!(times, @elapsed simulate_vm(case; nx, nv, dt, array_type = CuArray, method))
    end
    println(stderr, "case=$case nx=$nx nv=$nv dt=$dt runs=$runs method=$method mean_seconds=$(sum(times) / runs)")

    # Collect diagnostics separately so host transfers are excluded from timings.
    result = simulate_vm(case; nx, nv, dt, array_type = CuArray, diagnostics = true, method)
    println("time,magnetic_norm,ex_norm,ey_norm,total_energy,gauss_error")
    for row in result.history
        println(join(values(row), ','))
    end
end

main(ARGS)
