using FFTW
using LinearAlgebra
using Printf
using BenchmarkTools

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

        rhoh = vec(sum(f, dims = [1,2])) .* dvx .* dvy

        # Initialize arrays
        fnsvy = similar(f)
        fnp1vy = similar(f)

        # Loop over vy dimension
        for j = eachindex(vy)
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
        rhohvy = vec(sum(fnsvy, dims = [1,2])) .* dvx .* dvy
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

    while abs(t - tmax) > 1e-10
        iter += 1
        t += dt

        hf!(f, Ex, Ey, dt, vy, vy, kx)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

        #HE!(f, rx, ry, B, vx, vy, gx, gy, kx, kv, dt)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

        #HB!(f, Ey, B, vx, vy, kx, kvx, kvy, dt)
        gx, gy = updateg(rx, Ex, ry, Ey, x)

    end

end



# Run the simulation
@time results = vm_lie_1()
