export Mesh1D1V

abstract type AbstractMesh end

struct Mesh1D1V{T} <: AbstractMesh

    xmin::T
    xmax::T
    nx::Int
    vmin::T
    vmax::T
    nv::Int
    x::Vector{T}
    dx::T
    v::Vector{T}
    dv::T

    function Mesh1D1V(xmin::T, xmax::T, nx::Int, vmin::T, vmax::T, nv::Int) where {T}

        x = Vector{T}(LinRange(xmin, xmax, nx + 1)[1:(end - 1)])
        v = Vector{T}(LinRange(vmin, vmax, nv + 1)[1:(end - 1)])
        dx = (xmax - xmin) / nx
        dv = (vmax - vmin) / nv

        return new{T}(xmin, xmax, nx, vmin, vmax, nv, x, dx, v, dv)

    end

end

export Mesh1D2V

struct Mesh1D2V{T} <: AbstractMesh

    xmin::T
    xmax::T
    nx::Int
    vxmin::T
    vxmax::T
    nvx::Int
    vymin::T
    vymax::T
    nvy::Int
    x::Vector{T}
    dx::T
    vx::Vector{T}
    dvx::T
    vy::Vector{T}
    dvy::T

    function Mesh1D2V(
            xmin::T, xmax::T, nx::Int,
            vxmin::T, vxmax::T, nvx::Int,
            vymin::T, vymax::T, nvy::Int
        ) where {T}

        x = Vector{T}(LinRange(xmin, xmax, nx + 1)[1:(end - 1)])
        vx = Vector{T}(LinRange(vxmin, vxmax, nvx + 1)[1:(end - 1)])
        vy = Vector{T}(LinRange(vymin, vymax, nvy + 1)[1:(end - 1)])
        dx = (xmax - xmin) / nx
        dvx = (vxmax - vxmin) / nvx
        dvy = (vymax - vymin) / nvy

        return new{T}(
            xmin, xmax, nx, vxmin, vxmax, nvx, vymin, vymax, nvy,
            x, dx, vx, dvx, vy, dvy
        )

    end

end
