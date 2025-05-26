using FFTW
using LinearAlgebra
using VlasovAmpereEnergyPreserving
using Plots

# Grid parameters
Nx = 63
Nv = 29
tmax = 100

# Space mesh
kk = 0.5
xmin = 0
xmax = 2π/kk
dx = (xmax - xmin)/Nx
x = LinRange(xmin, xmax, Nx+1)[1:end-1]

# Wavenumbers
k = 2π/(xmax-xmin) * fftfreq(Nx, Nx)
kx = repeat(k, 1, Nv)'

# Velocity mesh
vmin = -14
vmax = 14
dv = (vmax - vmin)/Nv
v = range(vmin, stop = vmax-dv, length = Nv)
V = repeat(v, 1, Nx)
kkv = 2π/(vmax-vmin) * fftfreq(Nv, Nv)
kv = repeat(kkv, 1, Nx)

# Time mesh
dt = 0.5
t = 0
iter = 0
tabt = Float64[0]

# Initial condition
feq = 1/√(2π) * exp.(-0.5 * V .^ 2)
mass_hot = sum(feq) * dv
feq = feq / mass_hot[1]
fn = feq .* (1 .+ 0.001 * cos.(kk * X))

rho = sum(fn, dims = 1) * dv .- 1
E = zero(rho)
E = poisson(vec(rho), x, k)
r = √(0.5 * sum(E .^ 2) * dx)
g = E / r

@show size(E)

# Diagnostics
Ef = fft(E)
modes = abs.(Ef[2:10])
enelec = Float64[0.5*(dx*sum(E .^ 2))]
eelec = Float64[0.5*sum(E .^ 2)*dx]
r_history = Float64[r]
mom = Float64[sum(V .* fn)*dv*dx]
ekin = Float64[0.5*sum(V .^ 2 .* fn)*dv*dx]
etot = Float64[0.5*sum(V .^ 2 .* fn)*dv*dx+0.5*sum(E .^ 2)*dx]
diff_sav = Float64[0]
diff_poisson = Float64[0]

# Degree interpolation Lagrange
d = 2

# Method: meth=0 sav, meth=1 Lapenta
meth = 1

while abs(t) < abs(tmax)
    iter += 1
    t += dt
    mod(iter, 100) == 0 && println(t)

    fn, E = hf_lapenta(dt/2, dv, V, kx, k, fn, E)
    fn, E = HE_lapenta(fn, V, E, Nv, kv, dv, Nx, dx, dt, d)
    fn, E = hf_lapenta(dt/2, dv, V, kx, k, fn, E)

    # Update and diagnostics
    push!(tabt, iter * dt)
    E_diag = E
    Ef = fft(E_diag)
    rho = sum(fn, dims = 1) * dv .- 1
    Ep = poisson(vec(rho), x, k)
    @show size(E), size(Ep)
    push!(diff_poisson, norm(Ep - E))
    modes = abs.(Ef[2:10])
    push!(enelec, 0.5 * (dx * sum(E_diag .^ 2)))
    push!(r_history, r)
    push!(ekin, 0.5 * sum(V .^ 2 .* fn) * dv * dx)
    push!(mom, sum(V .* fn) * dv * dx)

    if meth == 0
        push!(etot, 0.5 * sum(V .^ 2 .* fn) * dv * dx + r^2)
        push!(diff_sav, r - √(0.5 * sum(E .^ 2) * dx))
    elseif meth == 1
        push!(etot, 0.5 * sum(V .^ 2 .* fn) * dv * dx + 0.5 * sum(E .^ 2) * dx)
    end
end

plot(tabt, ekin)
