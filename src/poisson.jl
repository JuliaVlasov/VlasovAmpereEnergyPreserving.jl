#=

function E=poisson(rho,x,k)

Nx=length(rho);
rhotmp=rho;
%x(Nx/2+1)=1;
k(1)=1;
Ef=fft(rhotmp)./(sqrt(-1)*k);
Ef(1)=0;
E=real(ifft(Ef));
=#

using FFTW

export poisson

function poisson(rho, x, k)
    Nx = length(rho)
    rhotmp = copy(rho)  # Make a copy to avoid modifying input

    # Perform FFT-based Poisson solve
    Ef = fft(rhotmp) 

    for i in 2:Nx
        Ef[i] = Ef[i] / (im * k[i])  # im is Julia's imaginary unit
    end

    Ef[1] = 0  # Set DC component to zero

    # Return real part of inverse FFT
    return real(ifft(Ef))
end
