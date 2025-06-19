using CUDA
using CUDA.CUDAKernels

# GPU-optimized kernel for Lagrange interpolation
function lagrange_interpolation_kernel(f, f_new, E, dt, dv, d, Nv)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= Nv
        shift = E * dt / dv
        j = floor(Int, i - shift)
        α = (i - shift) - j

        # Periodic boundary handling
        j = mod1(j, Nv)

        # Shared memory for stencil
        j_ind = @cuStaticSharedMem(Int32, (2d + 2,))
        f_vals = @cuStaticSharedMem(Float64, (2d + 2,))

        # Determine interpolation stencil indices
        for ii in (-d):(d + 1)
            if threadIdx().x == 1
                j_ind[ii + d + 1] = mod1(j + ii, Nv)
            end
        end
        sync_threads()

        # Load required values into shared memory
        if threadIdx().x <= 2d + 2
            f_vals[threadIdx().x] = f[j_ind[threadIdx().x]]
        end
        sync_threads()

        # Compute Lagrange coefficients
        L = 1.0
        f_new_i = 0.0

        @fastmath for ii in (-d):(d + 1)
            term = 1.0
            denom = 1.0

            for jj in (-d):(d + 1)
                if ii ≠ jj
                    term *= (α - jj)
                    denom *= (ii - jj)
                end
            end

            L = term / denom
            f_new_i += L * f_vals[ii + d + 1]
        end

        f_new[i] = f_new_i
    end
    return nothing
end

function semilag_gpu(f::CuArray{Float64}, E::Float64, dt::Float64, dv::Float64, d::Int)
    Nv = length(f)
    f_new = similar(f)

    # Kernel configuration
    threads = min(256, Nv)
    blocks = cld(Nv, threads)

    @cuda threads = threads blocks = blocks lagrange_interpolation_kernel(
        f, f_new, E, dt, dv, d, Nv
    )

    return f_new
end

# CPU-to-GPU wrapper function
function semilag(f::Vector{Float64}, Nv::Int, E::Float64, dt::Float64, dv::Float64, d::Int)
    # Transfer data to GPU
    f_gpu = CuArray(f)

    # Execute GPU kernel
    f_new_gpu = semilag_gpu(f_gpu, E, dt, dv, d)

    # Transfer result back to CPU
    return Array(f_new_gpu)
end

using CUDA

# Initialize data
Nv = 1024
f = rand(Nv)
E = 0.1
dt = 0.01
dv = 0.1
d = 2

# Warm-up CUDA (first call has compilation overhead)
semilag(f, Nv, E, dt, dv, d);

# Timed execution
@time f_new = semilag(f, Nv, E, dt, dv, d);

function semilag_multi_gpu(
        fs::Vector{Vector{Float64}}, Es::Vector{Float64}, dt::Float64, dv::Float64, d::Int
    )
    # Create CUDA streams
    streams = [CuStream() for _ in 1:length(fs)]

    # Allocate GPU arrays
    f_gpus = [CuArray(f) for f in fs]
    f_news = [similar(f_gpu) for f_gpu in f_gpus]

    # Launch kernels in parallel streams
    for i in 1:length(fs)
        Nv = length(fs[i])
        threads = min(256, Nv)
        blocks = cld(Nv, threads)
        @cuda threads = threads blocks = blocks stream = streams[i]
        lagrange_interpolation_kernel(f_gpus[i], f_news[i], Es[i], dt, dv, d, Nv)
    end

    # Synchronize and transfer results
    synchronize.(streams)
    return [Array(f_new) for f_new in f_news]
end
