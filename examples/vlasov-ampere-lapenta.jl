# -*- coding: utf-8 -*-
# # Vlasov-Ampere
#
# method : **SAV**

using FFTW
using LinearAlgebra
using Plots
using ProgressMeter
import VlasovAmpereEnergyPreserving: poisson!, semilag_d2
using DispersionRelations
import VlasovAmpereEnergyPreserving: hf_lapenta!, HE_lapenta!

function va_lapenta(nx, nv, tmax, dt, α)
    kk = 0.5
    xmin = 0
    xmax = 2π / kk
    dx = (xmax - xmin) / nx
    x = collect(LinRange(xmin, xmax, nx + 1)[1:(end - 1)])

    # Wavenumbers
    kx = collect(2π / (xmax - xmin) .* fftfreq(nx, nx))

    # Velocity mesh
    vmin = -14
    vmax = 14
    dv = (vmax - vmin) / nv
    v = collect(LinRange(vmin, vmax, nv + 1)[1:(end - 1)])
    kv = collect(2π / (vmax - vmin) .* fftfreq(nv, nv))

    # Time mesh
    t = 0
    iter = 0
    tabt = Float64[0]

    # Initial condition
    feq = 1 / √(2π) * exp.(-0.5 * v .^ 2)

    mass_hot = sum(feq) * dv
    feq = feq / mass_hot
    fn = feq .* (1 .+ α * cos.(kk * x'))
    rho = vec(sum(fn; dims = 1) * dv .- 1)
    E = zeros(nx)
    poisson!(E, rho, kx)
    r = √(0.5 * sum(E .^ 2) * dx)
    g = E / r

    # Diagnostics
    Ef = fft(E)
    modes = abs.(Ef[2:10])
    enelec = Float64[0.5 * (dx * sum(E .^ 2))]
    eelec = Float64[0.5 * sum(E .^ 2) * dx]
    r_history = Float64[r]
    mom = Float64[sum(v .* fn) * dv * dx]
    ekin = Float64[0.5 * sum(v .^ 2 .* fn) * dv * dx]
    etot = Float64[0.5 * sum(v .^ 2 .* fn) * dv * dx + 0.5 * sum(E .^ 2) * dx]
    diff_sav = Float64[0]
    diff_poisson = Float64[0]
    d = 2
    rhoh = vec(sum(fn; dims = 1)) .* dv

    fn = feq .* (1 .+ 0.001 * cos.(kk * x'))
    r = √(0.5 * sum(E .^ 2) * dx)
    g = E / r
    Ep = zeros(nx)

    while abs(t) < abs(tmax)
        iter += 1
        t += dt
        mod(iter, 100) == 0 && println(t)

        hf_lapenta!(fn, E, dt / 2, dv, v, kx)
        HE_lapenta!(fn, E, v, nv, kv, dv, nx, dx, dt)
        hf_lapenta!(fn, E, dt / 2, dv, v, kx)

        push!(tabt, iter * dt)
        push!(enelec, 0.5 * (dx * sum(E .^ 2)))
        Ef = fft(E)
        rho = vec(sum(fn; dims = 1)) * dv .- 1
        poisson!(Ep, rho, kx)
        push!(diff_poisson, norm(Ep - E))
        modes = abs.(Ef[2:10])
        push!(r_history, r)
        push!(ekin, 0.5 * sum(v .^ 2 .* fn) * dv * dx)
        push!(mom, sum(v .* fn) * dv * dx)
        push!(etot, 0.5 * sum(v .^ 2 .* fn) * dv * dx + r^2)
        push!(diff_sav, r - √(0.5 * sum(E .^ 2) * dx))
    end

    return tabt, enelec, etot, diff_poisson, fn
end

# +
nx = 63
nv = 129
tmax = 100
dt = 0.2
α = 0.001
@time tabt, enelec, etot, diff_poisson, fn = va_lapenta(nx, nv, tmax, dt, α)

plot(tabt, enelec; yaxis = :log)
line, ω, = fit_complex_frequency(tabt, enelec)
plot!(tabt, line; yaxis = :log)
title!("ω = $(imag(ω))")

# -
