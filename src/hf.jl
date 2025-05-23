function hf(dt, dvx, dvy, Nvx, Nvy, Nx, Vx, vy, kx, k, f, Ex, Ey)
    # solve the subsystem
    # df/dt + vx df/dvx = 0
    # dEx/dt = -Jx
    # dEy/dt = -Jy
    # dB/dt=drx/dt=dry/dt=0

    # Calculate initial charge densities
    rhoh = dropdims(sum(sum(f, dims=1), dims=2) * dvx * dvy
    
    # Initialize arrays
    fnsvy = similar(f)
    fnp1vy = similar(f)
    
    # Loop over vy dimension
    for j in 1:Nvy
        fs = dropdims(f[:,j,:], dims=2)  # Extract 2D slice
        fnsvy[:,j,:] = vy[j] * fs ./ Vx
        
        # Fourier transform and apply phase shift
        fs_hat = fft(fs', 1)  # FFT along x dimension
        fs_shifted = exp.(-im * Vx' .* kx * dt) .* fs_hat
        fs = real(ifft(fs_shifted, 1))'  # Inverse FFT
        
        f[:,j,:] = fs  # Update distribution function
        fnp1vy[:,j,:] = vy[j] * fs ./ Vx
    end

    # Calculate updated charge densities
    rhohvy = dropdims(sum(sum(fnsvy, dims=1), dims=2) * dvx * dvy)
    rhon = dropdims(sum(sum(f, dims=1), dims=2) * dvx * dvy)
    rhonvy = dropdims(sum(sum(fnp1vy, dims=1), dims=2) * dvx * dvy)

    # Compute current and update electric fields
    k_mod = copy(k)
    k_mod[1] = 1  # Avoid division by zero for DC component

    # Update Ex field
    rhs = (1 ./ (im * k_mod)) .* fft(rhon - rhoh)
    Ef = fft(Ex)
    Ef += rhs
    Ef[1] = 0  # Set DC component to zero
    Ex = real(ifft(Ef))

    # Update Ey field
    rhs = (1 ./ (im * k_mod)) .* fft(rhonvy - rhohvy)
    Ef = fft(Ey)
    Ef += rhs
    Ef[1] = 0  # Set DC component to zero
    Ey = real(ifft(Ef))

    return f, Ex, Ey
end
