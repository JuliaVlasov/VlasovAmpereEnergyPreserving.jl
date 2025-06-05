# -*- coding: utf-8 -*-
# # Vlasov-Ampere 
#
# method : **SAV**

using FFTW
using LinearAlgebra
using Plots
using ProgressMeter
import VlasovAmpereEnergyPreserving: poisson, semilag_d2
using DispersionRelations

function hf_lapenta!(f, E, dt, dv, v, kx)
    f_hat = fft(f, 2)
    f_hat .*= exp.(-im * v .* kx' * dt)
    f .= real(ifft(f_hat, 2))
end

function HE_lapenta!(
    f::Matrix{T},
    E::Vector{T},
    v::Vector{T},
    nv::Int,
    kv::Vector{T},
    dv::T,
    nx::Int,
    dx::T,
    dt::T,
) where {T<:AbstractFloat}

    # Pre-allocate outputs
    fn1 = similar(f)

    J = vec(sum(v .* f, dims = 1)) .* dv
    rho = vec(sum(f, dims = 1)) .* dv

    dt2_rho4 = (dt^2) .* rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2

    Enew = E ./ (1 .+ dt2_rho4) .- (dt * J) ./ (2 .+ dt2_rho2)

    # Update current and electric field
    Jnpdemi = J .+ 0.5 * dt .* Enew .* rho
    Enp1 = E .- dt .* Jnpdemi

    # Semi-Lagrangian step 
    for i in eachindex(Enew)
        ft = f[:, i]
        u1 = semilag_d2(ft, nv, Enew[i], dt, dv)
        f[:, i] .= u1
    end


    E .= Enp1

end

# +
function run(nx, nv, tmax, dt, α)

    kk = 0.5
    xmin = 0
    xmax = 2π/kk
    dx = (xmax - xmin)/nx
    x = LinRange(xmin, xmax, nx+1)[1:(end-1)] |> collect

    # Wavenumbers
    kx = collect(2π/(xmax-xmin) .* fftfreq(nx, nx))

    # Velocity mesh
    vmin = -14
    vmax = 14
    dv = (vmax - vmin)/nv
    v = LinRange(vmin, vmax, nv+1)[1:(end-1)] |> collect
    kv = collect(2π/(vmax-vmin) .* fftfreq(nv, nv))

    # Time mesh
    t = 0
    iter = 0
    tabt = Float64[0]

    # Initial condition
    feq = 1/√(2π) * exp.(-0.5 * v .^ 2)


    mass_hot = sum(feq) * dv
    feq = feq / mass_hot
    fn = feq .* (1 .+ α * cos.(kk * x'))
    rho = vec(sum(fn, dims = 1) * dv .- 1)
    E = poisson(rho, x, kx)
    r = √(0.5 * sum(E .^ 2) * dx)
    g = E / r

    # Diagnostics
    Ef = fft(E)
    modes = abs.(Ef[2:10])
    enelec = Float64[0.5*(dx*sum(E .^ 2))]
    eelec = Float64[0.5*sum(E .^ 2)*dx]
    r_history = Float64[r]
    mom = Float64[sum(v .* fn)*dv*dx]
    ekin = Float64[0.5*sum(v .^ 2 .* fn)*dv*dx]
    etot = Float64[0.5*sum(v .^ 2 .* fn)*dv*dx+0.5*sum(E .^ 2)*dx]
    diff_sav = Float64[0]
    diff_poisson = Float64[0]
    d = 2
    rhoh = vec(sum(fn, dims = 1)) .* dv

    fn = feq .* (1 .+ 0.001 * cos.(kk * x'))
    r = √(0.5 * sum(E .^ 2) * dx)
    g = E / r

    while abs(t) < abs(tmax)

        iter += 1
        t += dt
        mod(iter, 100) == 0 && println(t)

        hf_lapenta!(fn, E, dt/2, dv, v, kx)
        HE_lapenta!(fn, E, v, nv, kv, dv, nx, dx, dt)
        hf_lapenta!(fn, E, dt/2, dv, v, kx)

        push!(tabt, iter * dt)
        push!(enelec, 0.5 * (dx * sum(E .^ 2)))
        Ef = fft(E)
        rho = vec(sum(fn, dims = 1)) * dv .- 1
        Ep = poisson(rho, x, kx)
        push!(diff_poisson, norm(Ep - E))
        modes = abs.(Ef[2:10])
        push!(r_history, r)
        push!(ekin, 0.5 * sum(v .^ 2 .* fn) * dv * dx)
        push!(mom, sum(v .* fn) * dv * dx)
        push!(etot, 0.5 * sum(v .^ 2 .* fn) * dv * dx + r^2)
        push!(diff_sav, r - √(0.5 * sum(E .^ 2) * dx))

    end

    tabt, enelec, etot, diff_poisson, fn

end

# +
nx = 63
nv = 129
tmax = 100
dt = 0.2
α = 0.001
@time tabt, enelec, etot, diff_poisson, fn = run(nx, nv, tmax, dt, α)

plot(tabt, enelec, yaxis = :log)
line, ω, = fit_complex_frequency(tabt, enelec)
plot!(tabt, line, yaxis = :log)
title!("ω = $(imag(ω))")


# -
