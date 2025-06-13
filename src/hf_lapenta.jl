#=
function [ f,E ] = hf_lapenta(dt,dv,V,kx,k,f,E)

    f=real(ifft(exp(-sqrt(-1)*V'.*kx*dt).*fft(f')))';

end
=#

export hf_lapenta

function hf_lapenta!(f, E, dt, dv, v, kx)
    f_hat = fft(f, 2)
    f_hat .*= exp.(-im * v .* kx' * dt)
    f .= real(ifft(f_hat, 2))
end

