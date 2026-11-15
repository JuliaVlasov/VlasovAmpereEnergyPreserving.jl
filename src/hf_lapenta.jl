export hf_lapenta

@doc raw"""
    hf_lapenta!(f, E, dt, dv, v, kx)

Advance ``\partial_t f+v\partial_xf=0`` on a periodic spatial mesh by
applying the Fourier multiplier ``e^{-ikv\Delta t}``.
Update `f`, of size `(length(v), length(kx))`, in place and return it.
`E` is unchanged; `dv` is retained for compatibility with the calling interface.
"""
function hf_lapenta!(f, E, dt, dv, v, kx)
    f_hat = fft(f, 2)
    f_hat .*= exp.(-im * v .* kx' * dt)
    return f .= real(ifft(f_hat, 2))
end
