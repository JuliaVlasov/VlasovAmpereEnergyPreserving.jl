using FFTW
using LinearAlgebra
using Printf
import VlasovAmpereEnergyPreserving: updateg, poisson

# Hongtao-Lapenta splitting

# Computational domain parameters
nx = 16
nvx = 47
nvy = 47
dt = 0.125 / 2
tmax = 300.0
nt = floor(Int, tmax / dt)

# Test case selection
kinit = 1.25
vxmin, vxmax = -0.2, 0.2
vymin, vymax = -0.5, 0.5

# Initialize arrays
finit = zeros(nvx, nvy, nx)
f = zeros(nvx, nvy, nx)
f1 = zeros(nvx, nvy, nx)
f2 = zeros(nvx, nvy, nx)
f3 = zeros(nvx, nvy, nx)
tr_mat = zeros(nvx, nvy, nx)
diag = zeros(10, nt+1)
diag_tmp = zeros(10)
screen_var = zeros(3)


xmin, xmax = 0, 2π / kinit
dx = (xmax-xmin) / nx
x = LinRange(xmin, xmax, nx+1)[1:end-1]

kx = 2π/(xmax-xmin) * fftfreq(nx, nx) |> collect

dvx = 2 * vxmax / nvx
dvy = 2 * vymax / nvy
vx = LinRange(vxmin, vxmax, nvx+1)[1:end-1]
vy = LinRange(vymin, vymax, nvy+1)[1:end-1]

f0 = zeros(nvx, nvy, nx)

α = 0.0001
σ1 = 0.02/sqrt(2)
σ2 = sqrt(12)*σ1
for i = eachindex(vx), j = eachindex(vy), k = eachindex(x)
    f0[i, j, k] = exp(-0.5*vx[i]^2/σ1^2) * exp(-0.5*vy[j]^2/σ2^2) * (1 + α*cos(kinit*x[k])) /(2π*σ1*σ2) 
end

Ex = zeros(nx)
Ey = zeros(nx)
B = zeros(nx)
rho = vec(sum(f0, dims = (1, 2))) * dvx * dvy
Ex = poisson(rho, x, kx)
B = α * cos.(kinit .* x)

f = copy(f0)

rho0 = sum(f) * dvx * dvy
Ex0 = copy(Ex)
B0 = copy(B)
Ey0 = copy(Ey)


rx = sqrt(0.5 * sum(Ex .^ 2) * dx + 1.0)
ry = sqrt(0.5 * sum(Ey .^ 2) * dx + 1.0)
gx, gy = updateg(rx, Ex, ry, Ey, x, nx)

# Useful arrays
kvx = 2π / (vxmax - vxmin) * fftfreq(nvx, nvx) |> collect
kvy = 2π / (vymax - vymin) * fftfreq(nvy, nvy) |> collect

#=

# Initial diagnostics
diag_tmp = compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, nx, nvx, nvy, dx, 0.0)
en_init = diag_tmp[7]
diag[:, 1] = diag_tmp
verif_poisson = [0.0]
=#

t = 0.0
iter = 0

function hf_lapenta2d!(f, dt, nvy, vx, kx)

    for j=axes(f, 2)

        fs = fft(f[:,j,:], 1)
        fs .*= exp.(-1im * vx .* kx' * dt)
        fs .= ifft(fs, 1)

        f[:,j,:] .= real(fs)
    end
    
end


while abs(t - tmax) > 1e-10
    global iter
    iter += 1
    global t
    t += dt

    if mod(iter, 50) == 0
        screen_var[1] = t
        @printf("Time: %.2f, Max gy: %.4e\n", t, maximum(abs.(gy)))
    end

    hf_lapenta2d!(f, dt, nvy, vx, kx)

#=
    f, Ex, Ey = HE_lapenta2d(f, vx, vy, Ex, Ey, nvx, nvy, dvx, dvy, nx, dx, dt, d)
    f, Ex, Ey, B = HB_lapenta2d(
        f,
        Ex,
        Ey,
        B,
        vx,
        vy,
        kk,
        kkvx,
        kkvy,
        nvx,
        nvy,
        dvx,
        dvy,
        nx,
        dt,
        dx,
        d,
        meth,
    )
=#

    ## Diagnostics
    #diag_tmp = compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, nx, nvx, nvy, dx, t)
    #rho = sum(f) * dvx * dvy
    #Ex_poisson = poisson(dropdims(sum(f, dims = (1, 2)), x, kk))
    #push!(verif_poisson, norm(Ex_poisson - Ex))
    #diag[:, iter+1] = diag_tmp
end
