# -*- coding: utf-8 -*-
# # Vlasov-Ampere 
#
# method : **SAV**

using FFTW
using LinearAlgebra
using Plots
using ProgressMeter

# Grid parameters
Nx = 63
Nv = 29
tmax = 1000

# Space mesh
kk = 0.5
xmin = 0
xmax = 2π/kk
dx = (xmax - xmin)/Nx
x = LinRange(xmin, xmax, Nx+1)[1:end-1] |> collect

# Wavenumbers
kx = collect(2π/(xmax-xmin) .* fftfreq(Nx, Nx))

# Velocity mesh
vmin = -14
vmax = 14
dv = (vmax - vmin)/Nv
v = LinRange(vmin, vmax, Nv+1)[1:end-1] |> collect
kv = collect(2π/(vmax-vmin) .* fftfreq(Nv, Nv))

# Time mesh
dt = 0.5
t = 0
iter = 0
tabt = Float64[0]

# Initial condition
feq = 1/√(2π) * exp.(-0.5 * v .^ 2)
plot(v, feq)

mass_hot = sum(feq) * dv
feq = feq / mass_hot[1]
fn = feq .* (1 .+ 0.001 * cos.(kk * x'))
surface(x, v, fn)

rho = vec(sum(fn, dims = 1) * dv .- 1)
plot(x, rho)

E = zero(rho)
E = poisson(rho, x, kx)
plot(x, E)

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

# Degree interpolation Lagrange

d = 2

rhoh = vec(sum(fn, dims=1)) .* dv  

# +
import VlasovAmpereEnergyPreserving.hf

function hf(dt::Float64, dv::Float64, v::Vector{Float64}, kx::Vector{Float64}, f::Matrix{Float64}, E::Vector{Float64})

    rhoh = vec(sum(f, dims=1)) .* dv  
    ft = fft(fn, 2)
    ft .* exp.(-1im * v .* kx' * dt )
    f .= real(ifft(ft, 2))
    rhon = vec(sum(f, dims=1)) .* dv
    k_mod = kx
    k_mod[1] = 1  
    
    rhs = fft(rhon .- rhoh) ./ (1im * k_mod)
    Ef = fft(E)
    Ef .+= rhs 
    Ef[1] = 0 
    E = real(ifft(Ef))
    f, E
end

# -

@inline function semilag(f::Vector{Float64}, Nv::Int, E::Float64,
                        dt::Float64, dv::Float64, d::Int)::Vector{Float64}
    # Initialize output and working arrays
    f_new = similar(f)
    alpha = zeros(2d+2)
    j_ind = Vector{Int}(undef, 2d+2)

    # Precompute the shift amount
    shift = E * dt / dv

    # Main loop over velocity grid points
    @inbounds for i in 1:Nv
        # Compute base index and fractional part
        j = floor(Int, i - shift)
        α = (i - shift) - j

        # Periodic boundary handling (optimized)
        j = mod1(j, Nv)

        # Determine interpolation stencil indices
        @simd for ii in -d:(d+1)
            j_ind[ii+d+1] = mod1(j + ii, Nv)
        end

        # Compute Lagrange coefficients
        P = ones(2d+2, 2d+2)
        C = ones(2d+2, 2d+2)

        @fastmath for ii in -d:(d+1), jj in -d:(d+1)
            if ii ≠ jj
                P[jj+d+1, ii+d+1] = (α - ii)
            end
            C[jj+d+1, ii+d+1] = (jj - ii)
        end

        # Compute Lagrange weights
        L = zeros(2d+2)
        @fastmath for ii in -d:(d+1)
            prodP = 1.0
            prodC = 1.0
            for jj in -d:(d+1)
                if jj ≠ ii
                    prodP *= P[ii+d+1, jj+d+1]
                    prodC *= C[ii+d+1, jj+d+1]
                end
            end
            L[ii+d+1] = prodP / prodC
        end

        # Apply interpolation
        f_new[i] = zero(Float64)
        @simd for ii in 1:(2d+2)
            f_new[i] += L[ii] * f[j_ind[ii]]
        end
    end

    return f_new
end

function HE(f, v, r, g, Nv, kv, dv, Nx, dx, dt, d)


    J = vec(sum(v .* f, dims=1 )) .* dv 
    rho = vec(sum(f, dims=1)) .* dv
    Jnpdemi = J .+ 0.25 * dt * r * g .* rho
    rnp1 = (1/(1+(dt*dt/8) * sum(rho .* g.^2) * dx)) * (r-0.5*dt * sum(g .* Jnpdemi) * dx)
    
    Enew = 0.5 * (rnp1+r) * g 
    
    for i=eachindex(Enew)
        ft = f[:,i]
        u1 = semilag(ft, Nv, Enew[i], dt, dv, d)
        f[:,i] .= u1
    end
        
    r = rnp1
    
    return f , r
end

while abs(t) < abs(tmax)
    iter += 1
    t += dt
    mod(iter, 100) == 0 && println(t)

    
    fn, E = hf(dt/2, dv, v, kx, fn, E)
    fn, r = HE(fn, v, r, g, Nv, kv, dv, Nx, dx, dt, d)
    fn, E = hf(dt/2, dv, v, kx, fn, E)
    
    # Update and diagnostics
    push!(tabt, iter * dt)
    E_diag = E
    Ef = fft(E_diag)
    rho = vec(sum(fn, dims = 1) * dv .- 1)
    Ep = poisson(rho, x, kx)
    push!(diff_poisson, norm(Ep - E))
    modes = abs.(Ef[2:10])
    push!(enelec, 0.5 * (dx * sum(E_diag .^ 2)))
    push!(r_history, r)
    push!(ekin, 0.5 * sum(v .^ 2 .* fn) * dv * dx)
    push!(mom, sum(v .* fn) * dv * dx)

    push!(etot, 0.5 * sum(v .^ 2 .* fn) * dv * dx + r^2)
    push!(diff_sav, r - √(0.5 * sum(E .^ 2) * dx))
end


plot(tabt, enelec)


