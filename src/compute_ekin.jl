#=
function kin = compute_ekin(f,vx,vy,Nx,Nvx,Nvy,dx)

    dvx=vx(2)-vx(1);dvy=vy(2)-vy(1);
    for k=1:Nx
        kin(k)=0;
        for i=1:Nvx-1
            for j=1:Nvy-1
                normv=0.5*(vx(i)*vx(i)+vy(j)*vy(j));
                kin(k)=kin(k)+normv*f(i,j,k)*dvx*dvy;
            end
        end
    end

end
=#

using CUDA
using KernelAbstractions

function compute_ekin_gpu(
    f::CuArray{Float64,3}, vx::CuArray{Float64}, vy::CuArray{Float64}, dx::Float64
)::CuArray{Float64}
    Nx = size(f, 3)
    Nvx = length(vx)
    Nvy = length(vy)
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]
    kin = CUDA.zeros(Float64, Nx)

    # Kernel for GPU computation
    function kernel!(kin, f, vx, vy, dvx, dvy)
        k = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        i = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        j = threadIdx().z + (blockIdx().z - 1) * blockDim().z

        @inbounds if k <= size(f, 3) && i <= size(f, 1) && j <= size(f, 2)
            normv = 0.5f0 * (vx[i]^2 + vy[j]^2)
            CUDA.atomic_add!(pointer(kin, k), normv * f[i, j, k] * dvx * dvy)
        end
        return nothing
    end

    # Launch configuration
    threads = (16, 16, 1)
    blocks = (cld(Nx, threads[1]), cld(Nvx, threads[2]), cld(Nvy, threads[3]))

    @cuda threads=threads blocks=blocks kernel!(kin, f, vx, vy, dvx, dvy)
    synchronize()

    return kin .* dx  # Multiply by dx after reduction
end

function compute_ekin(
    f::AbstractArray{Float64,3},
    vx::AbstractVector{Float64},
    vy::AbstractVector{Float64},
    dx::Float64;
    device::Symbol=:cpu,
)
    if device == :gpu && CUDA.functional()
        f_gpu = CuArray(f)
        vx_gpu = CuArray(vx)
        vy_gpu = CuArray(vy)
        return Array(compute_ekin_gpu(f_gpu, vx_gpu, vy_gpu, dx))
    else
        return compute_ekin_cpu(f, vx, vy, dx)
    end
end

using Base.Threads

function compute_ekin_cpu(
    f::Array{Float64,3}, vx::Vector{Float64}, vy::Vector{Float64}, dx::Float64
)::Vector{Float64}
    Nx = size(f, 3)
    Nvx = length(vx)
    Nvy = length(vy)
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]
    kin = zeros(Float64, Nx)

    @threads for k in 1:Nx
        kin_local = 0.0
        @inbounds for j in 1:(Nvy - 1)
            vy2 = 0.5 * vy[j]^2
            for i in 1:(Nvx - 1)
                normv = 0.5 * vx[i]^2 + vy2
                kin_local += normv * f[i, j, k] * dvx * dvy
            end
        end
        kin[k] = kin_local
    end

    return kin .* dx
end
