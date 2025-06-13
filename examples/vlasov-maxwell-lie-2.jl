
using FFTW
using LinearAlgebra
using Printf
using BenchmarkTools
using .Threads
import VlasovAmpereEnergyPreserving: poisson, updateg, semilag_d2, compute_diag

"""
solve the subsystem 
df/dt + rxgx df/dvx + rygydf/dvy = 0
drx/dt=-0.5*int (gx Jx) dx
dry/dt=-0.5*int (gy Jy) dx
dEx/dt=dEy/dt=dB/dt =0
"""

function HE2(f, vx, vy, rx, gx, Ex, ry, gy, Ey, dx, dt)


    nvx, nvy, nx = size(f)

    Jx = zeros(nx)
    Jy = zeros(ny)
    rho = zeros(nx)

    for i = 1:nx
        Jx[i]=0.0
        Jy[i]=0.0
        rho[i]=0.0;
        for j = 1:nvx, k = 1:nvy
            Jx[i] += vx(j)*f[j, k, i]
            Jy[i] += vy(k)*f[j, k, i]
            rho[i] += fn[j, k, i]
        end
    end
    Jx .*= dvx * dvy
    Jy .*= dvx * dvy
    rho .*= dvx * dvy

    gx, gy = updateg(rx, Ex, ry, Ey, x, nx)

    denom=1+(dt*dt/8.0)*sum(rho .* gx .^ 2)*dx;
    rxnp1=(rx-0.5*dt*sum(gx .* Jx)*dx-(dt*dt/8.0)*rx*sum(rho .* gx .^ 2)*dx)/denom;
    rxdemi=rx/denom-(0.25*dt/denom)*sum(gx .* Jx)*dx;
    Exnew=rxdemi*gx;

    denom=1+(dt*dt/8.0)*sum(rho .* gy .^ 2)*dx;
    rynp1=(ry-0.5*dt*sum(Jy .* gy)*dx-(dt*dt/8.0)*ry*sum(rho .* gy .^ 2)*dx)/denom;
    rydemi=ry/denom-(0.25*dt/denom)*sum(gy .* Jy)*dx;
    Eynew=rydemi*gy;

    for j = 1:nvy, i = 1:nx
        ft=f[:, j, i]
        u1=semilag_d2(ft, nvx, Exnew[i], dt, dvx)
        f[:, j, i] .= u1
    end


    for j = 1:nvx, i = 1:nx
        ft = fn[j, :, i]
        u1 = semilag(ft, nvy, Eynew[i], dt, dvy)
        f[j, :, i] .= u1
    end

    rxnp1, rynp1

end


"""
solve the subsystem 
df/dt + B Jv . df/dv = 0
dB/dt = -ry d(gy)/dx 
dEy/dt=-dB/dx
dry/dt=-0.5*int (gy dB/dx)) dx = 0.5*int (d(gy)/dx B) dx 
dEx/dt=0
"""
function HB2!(f, vx, vy, Ey, ry, gy, B, kx, kvx, kvy, dt)

    for i = 2:(nx-1)
        dxgy[i]=(gy[i+1]-gy[i-1]) / (2dx)
    end
    dxgy[1]=(gy[2]-gy[nx])/(2dx)
    dxgy[nx]=(gy[1]-gy[nx-1])/(2dx)

    denom=1+(dt*dt/8.0)*sum(dxgy .* dxgy)*dx
    rynp1=(ry+0.5*dt*sum(B .* dxgy)*dx-(dt*dt/8.0)*ry*sum(dxgy .* dxgy)*dx)/denom;
    rydemi=ry/denom+0.25*dt*sum(B .* dxgy)*dx/denom;

    Bnp1=B-dt*rydemi*dxgy
    Bdemi=B-0.5*dt*rydemi*dxgy
    for i = 2:(nx-1)
        dxBdemi[i]=(Bdemi[i+1]-Bdemi[i-1])/(2dx)
    end
    dxBdemi[1]=(Bdemi[2]-Bdemi[nx])/(2dx)
    dxBdemi[nx]=(Bdemi[1]-Bdemi[nx-1])/(2dx)

    Eynp1=Ey-dt*dxBdemi

    fn=fft(fn, 1);
    for l = 1:nx, j = 1:nvy, i = 1:nvx
        fn[i, j, l]=exp(-1im*kkvx[i]*vy[j]*tan(Bdemi[l]*dt/2)) * fn[i, j, l]
    end
    fn=ifft(fn, 1)
    fn=fft(fn, 2)
    for l = 1:nx, i = 1:nvx, j = 1:nvy
        fn[i, j, l]=exp(1im*kvy[j]*vx[i]*sin(Bdemi[l]*dt)) * fn[i, j, l]
    end
    fn=ifft(fn, [], 2)

    fn=fft(fn, [], 1)
    for l = 1:nx, j = 1:nvy, i = 1:nvx
        fn[i, j, l]=exp(-1im*kvx[i]*vy[j]*tan(Bdemi[l]*dt/2)) .* fn[i, j, l]
    end

    fn=ifft(fn, 1)

    Ey .= Eynp1
    ry .= rynp1
    B .= Bnp1

end

function vm_lie_2(tmax)

    nx = 16
    nvx = 47
    nvy = 47
    dt = 0.125 / 2
    nt = floor(Int, tmax / dt)

    kinit = 1.25
    xmin, xmax = 0, 2π / kinit
    vxmin, vxmax = -0.2, 0.2
    vymin, vymax = -0.5, 0.5

    f = zeros(nvx, nvy, nx)

    x = LinRange(xmin, xmax, nx+1)[1:(end-1)]
    kx = 2π / (xmax - xmin) * collect(fftfreq(nx, nx))
    vx = LinRange(vxmin, vxmax, nvx+1)[1:(end-1)]
    vy = LinRange(vymin, vymax, nvy+1)[1:(end-1)]

    α = 0.0001
    σ1 = 0.02/sqrt(2)
    σ2 = sqrt(12)*σ1
    for i in eachindex(vx), j in eachindex(vy), k in eachindex(x)
        f[i, j, k] =
            exp(-0.5*vx[i]^2/σ1^2) * exp(-0.5*vy[j]^2/σ2^2) * (1 + α*cos(kinit*x[k])) /
            (2π*σ1*σ2)
    end

    Ex = zeros(nx)
    Ey = zeros(nx)
    B = zeros(nx)
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]
    rho = vec(sum(f, dims = (1, 2))) * dvx * dvy
    Ex .= poisson(rho, x, kx)
    B = α * cos.(kinit .* x)

    kvx = 2π / (vxmax - vxmin) * fftfreq(nvx, nvx) |> collect
    kvy = 2π / (vymax - vymin) * fftfreq(nvy, nvy) |> collect

    dx = x[2] - x[1]
    rx=sqrt(0.5*sum(Ex .* Ex)*dx+1.0)
    ry=sqrt(0.5*sum(Ey .* Ey)*dx+1.0)

    gx, gy = updateg(Ex, Ey, rx, ry)

    t = 0.0
    iter = 0


    diags = [compute_diag(f, Ex, Ey, B, vx, vy, rx, ry, dx, t)]

    while abs(t - tmax) > 1e-10
        iter += 1
        t += dt

        hf!(f, Ex, Ey, dt, vx, vy, kx)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        HB!(f, Ey, B, vx, vy, kx, kvx, kvy, dt)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

        rx, ry = HE2!(f, Ex, Ey, gx, gy, rx, ry, x, vx, vy, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        rx, ry = HB2!(f, Ex, Ey, B, gx, gy, rx, ry, kx, kvx, kvy, dt)
        updateg!(gx, gy, Ex, Ey, rx, ry)

        push!(diags, compute_diag(f, Ex, Ey, B, vx, vy, rx, ry, dx, t))

    end

    return diags

end


# Run the simulation
tmax = 100.0
@time lie2 = vm_lie_2(tmax)
