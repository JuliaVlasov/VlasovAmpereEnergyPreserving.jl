using FFTW

export poisson

function poisson(rho, x, kx)
    nx = length(rho)
    Ef = fft(rho)
    for i in 2:nx
        Ef[i] = Ef[i] / (1im * kx[i])
    end

    Ef[1] = 0

    return real(ifft(Ef))
end

function poisson!(Ex, rho, kx)
    nx = length(Ex)
    Ef = fft(rho)
    for i in 2:nx
        Ef[i] = Ef[i] / (1im * kx[i])
    end
    Ef[1] = 0
    Ex .= real(ifft(Ef))
end
