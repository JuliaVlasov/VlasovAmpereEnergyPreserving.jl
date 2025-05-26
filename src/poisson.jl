using FFTW

export poisson

function poisson(rho, x, kx)
    nx = length(rho)
    rhotmp = copy(rho) 

    Ef = fft(rhotmp) 

    for i in 2:nx
        Ef[i] = Ef[i] / (im * kx[i])
    end

    Ef[1] = 0  

    return real(ifft(Ef))
end
