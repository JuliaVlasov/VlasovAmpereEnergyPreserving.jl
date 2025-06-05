# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Julia 1.11.5
#     language: julia
#     name: julia-1.11
# ---

# Semi-Lagrangian function 

using FFTW
using Plots
import VlasovAmpereEnergyPreserving: semilag



# +
Nx = 63
Nv = 63

xmin = -14
xmax = 14
dx = (xmax - xmin)/Nx
x = LinRange(xmin, xmax, Nx+1)[1:(end-1)] |> collect
vmin = -14
vmax = 14
dv = (vmax - vmin)/Nv
v = LinRange(vmin, vmax, Nv+1)[1:(end-1)] |> collect


# -

f = exp.(-0.5 * (v .- 5) .^ 2) .* exp.(-0.5 * x .^ 2)'
@gif for j = 1:100
    e = 0.1 .* ones(Nx)
    d = 2
    dt = 0.5
    for i in eachindex(v)
        ft = f[:, i]
        f[:, i] .= semilag(ft, Nv, v[i], tan(dt/2), dv, d)
    end
    for i in eachindex(x)
        ft = f[i, :]
        f[i, :] .= semilag(ft, Nv, -x[i], sin(dt), dx, d)
    end
    for i in eachindex(v)
        ft = f[:, i]
        f[:, i] .= semilag(ft, Nv, v[i], tan(dt/2), dv, d)
    end
    plot(x, v, f, aspect_ratio = 1)
    xlims!(-15, 15)
    ylims!(-15, 15)
end
