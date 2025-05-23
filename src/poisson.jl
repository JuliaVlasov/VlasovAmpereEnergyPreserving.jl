using FFTW

export poisson

function poisson(rho, x, k)
    Nx = length(rho)
    rhotmp = copy(rho)  # Make a copy to avoid modifying input
    
    # Perform FFT-based Poisson solve
    Ef = fft(rhotmp) ./ (im * k)  # im is Julia's imaginary unit
    Ef[1] = 0  # Set DC component to zero
    
    # Return real part of inverse FFT
    return real(ifft(Ef))
end
