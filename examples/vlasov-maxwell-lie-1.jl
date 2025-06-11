using FFTW
using LinearAlgebra
using Printf
using BenchmarkTools
using .Threads

function vm_lie_1()

    nx = 16
    nvx = 47
    nvy = 47
    dt = 0.125 / 2
    tmax = 300.0
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
    Ex = poisson(rho, x, kx)
    B = α * cos.(kinit .* x)

    kvx = 2π / (vxmax - vxmin) * fftfreq(nvx, nvx) |> collect
    kvy = 2π / (vymax - vymin) * fftfreq(nvy, nvy) |> collect

    dx = x[2] - x[1]
    rx=sqrt(0.5*sum(Ex .* Ex)*dx+1.0)
    ry=sqrt(0.5*sum(Ey .* Ey)*dx+1.0)

    gx, gy = updateg(rx, Ex, ry, Ey, x)

    t = 0.0
    iter = 0


    """
    solve the subsystem

    df/dt + vx df/dvx = 0
    dEx/dt = -Jx
    dEy/dt = -Jy
    dB/dt=drx/dt=dry/dt=0
    """
    function hf!(f, Ex, Ey, dt, vx, vy, kx)

        rhoh = vec(sum(f, dims = [1, 2])) .* dvx .* dvy

        # Initialize arrays
        fnsvy = similar(f)
        fnp1vy = similar(f)

        # Loop over vy dimension
        for j in eachindex(vy)
            fs = f[:, j, :]  # Extract 2D slice
            fnsvy[:, j, :] = vy[j] * fs ./ vx

            # Fourier transform and apply phase shift
            fs_hat = fft(fs', 1)  # FFT along x dimension
            fs_shifted = exp.(-im * vx' .* kx * dt) .* fs_hat
            fs = real(ifft(fs_shifted, 1))'  # Inverse FFT

            f[:, j, :] .= fs  # Update distribution function
            fnp1vy[:, j, :] .= vy[j] * fs ./ vx
        end

        # Calculate updated charge densities
        rhohvy = vec(sum(fnsvy, dims = [1, 2])) .* dvx .* dvy
        rhon = vec(sum(f, dims = [1, 2])) .* dvx .* dvy
        rhonvy = vec(sum(f, dims = [1, 2])) .* dvx .* dvy

        # Compute current and update electric fields
        k_mod = copy(kx)
        k_mod[1] = 1  # Avoid division by zero for DC component

        # Update Ex field
        rhs = (1 ./ (im * k_mod)) .* fft(rhon .- rhoh)
        Ef = fft(Ex)
        Ef .+= rhs
        Ef[1] = 0  # Set DC component to zero
        Ex .= real(ifft(Ef))

        # Update Ey field
        rhs = (1 ./ (im * k_mod)) .* fft(rhonvy - rhohvy)
        Ef = fft(Ey)
        Ef .+= rhs
        Ef[1] = 0  # Set DC component to zero
        Ey .= real(ifft(Ef))

    end



    """
    solve the subsystem 
    df/dt + rxgx df/dvx + rygydf/dvy = 0
    dB/dt = -d(ry gy)/dx 
    drx/dt=-0.5*int (gx Jx) dx
    dry/dt=-0.5*int (gy (Jy+dB/dx)) dx
    dEx/dt=dEy/dt=0
    """
    function HE!(fn, rx, ry, B, vx, vy, gx, gy, kx, dt)


        for i = 1:nx
            Jx(i)=0.0;
            Jy(i)=0.0;
            rho(i)=0;
            for j = 1:nvx, k = 1:nvy
                Jx(i)=Jx(i)+vx(j)*fn(j, k, i);
                Jy(i)=Jy(i)+vy(k)*fn(j, k, i);
                rho(i)=rho(i)+fn(j, k, i);
            end
        end
        Jx=Jx*dvx*dvy;
        Jy=Jy*dvx*dvy;
        rho=rho*dvx*dvy;

        Jxnpdemi=Jx+0.25*dt*rx*gx .* rho;
        denom=1+(dt*dt/8)*sum(rho .* gx .^ 2)*dx;
        rxnp1=(rx-0.5*dt*sum(gx .* Jx)*dx-(dt*dt/8)*rx*sum(rho .* gx .^ 2)*dx)/denom;
        Exnew=0.5*(rxnp1+rx)*gx;

        Jynpdemi=Jy+0.25*dt*ry*gy .* rho;
        dxgy=real(ifft(sqrt(-1)*kk .* fft(gy)));
        dx2gy=real(ifft(-kk .* kk .* fft(gy)));
        dxB=real(ifft(sqrt(-1)*kk .* fft(B)));
        denom=1+(dt*dt/8)*sum(rho .* gy .^ 2)*dx+(dt*dt/8)*sum(dxgy .* dxgy)*dx;
        rynp1=(
            ry-0.5*dt*sum(
                Jy .* gy,
            )*dx+0.5*dt*sum(
                B .* dxgy,
            )*dx-(dt*dt/8)*ry*sum(rho .* gy .^ 2)*dx-(dt*dt/8)*ry*sum(dxgy .* dxgy)*dx
        )/denom;

        Eynew=0.5*(rynp1+ry)*gy;

        faraday=real(ifft(sqrt(-1)*kk .* fft(Eynew)));

        B=B-dt*faraday;

        for j = 1:nvy, i = 1:nx
            ft = fn[:, j, i]
            u1 = semilag_d2(ft, nvx, Exnew[i], dt, dvx)
            fn[:, j, i] .= u1
        end


        for j = 1:nvx, i = 1:nx
            ft=fn[j, :, i]
            u1=semilag(ft, nvy, Eynew(i), dt, dvy, d);
            fn[j, :, i] .= u1
        end

        rx=rxnp1;
        ry=rynp1;

    end


    """
    solve the subsystem 
    df/dt+B Jv.df/dv=0 (with semi-Lagrangian + exact splitting) 
    dEy/dt=-dB/dx
    dB/dt=drx/dt=dry/dt=0
    """
    function HB!(f, Ey, B, vx, vy, kx, kvx, kvy, dt)

        fn = fft(f, 1)
        for l = 1:nx, j = 1:nvy, i = 1:nvx
            fn[i, j, l]=exp(-1im*kvx[i]*vy[j]*tan(B[l]*dt/2)) * fn[i, j, l]
        end
        fn .= ifft(fn, 1)

        fn .= fft(fn, 2)
        for l = 1:nx, i = 1:nvx, j = 1:nvy
            fn[i, j, l]=exp(1im*kkvy(j)*vx(i)*sin(B(l)*dt)) * fn[i, j, l];
        end
        fn .= ifft(fn, 2)

        fn .= fft(fn, 1)
        for l = 1:nx, j = 1:nvy, i = 1:nvx
            fn[i, j, l]=exp(-1im*kkvx(i)*vy(j)*tan(B(l)*dt/2)) * fn[i, j, l]
        end
        fn .= ifft(fn, 1)

        amperey=real(ifft(1im * kk .* fft(B)))
        Ey .= Ey .- dt .* amperey
    end

    while abs(t - tmax) > 1e-10
        iter += 1
        t += dt

        hf!(f, Ex, Ey, dt, vy, vy, kx)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

        rx, ry = HE!(f, rx, ry, B, vx, vy, gx, gy, kx, dt)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

        HB!(f, Ey, B, vx, vy, kx, kvx, kvy, dt)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

    end

end



# Run the simulation
@time results = vm_lie_1()
