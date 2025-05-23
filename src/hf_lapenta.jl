#=
function [ f,E ] = hf_lapenta(dt,dv,V,kx,k,f,E)


    f=real(ifft(exp(-sqrt(-1)*V'.*kx*dt).*fft(f')))';

end
=#

using FFTW

export hf_lapenta

function hf_lapenta(dt, dv, V, kx, k, f, E)
    # Compute Fourier transform along x-dimension (transpose first)
    f_hat = fft(f', 1)

    @show size(f_hat)
    @show size(V)
    @show size(kx)
    # Apply phase shift in Fourier space
    phase_shift = exp.(-im * V .* kx * dt)
    f_shifted = phase_shift' .* f_hat

    # Inverse Fourier transform and transpose back
    f = real(ifft(f_shifted, 1))'

    return f, E
end

function hf_lapenta!(f, E, dt, dv, V, kx, k)
    f_hat = fft(f', 1)
    f_hat .*= exp.(-im * V' .* kx * dt)
    f .= real(ifft(f_hat, 1))'
    return f, E
end
