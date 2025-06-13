"""
solve the subsystem 

```math
df/dt + vx df/dx = 0 
dEx/dt = 0
dEy/dt = 0
dB/dt  = 0
```

"""

function hf_lapenta2d!(
    f::AbstractArray{T,3}, vx::AbstractVector, kx::AbstractVector, dt::Real
) where {T}
    nvx, nvy, nx = size(f)

    fs = complex(zeros(nvx, nx))

    for j in axes(f, 2)
        fs .= f[:, j, :]
        fs .= fft(fs, 2)
        fs .*= exp.(-1im .* vx .* kx' * dt)
        fs .= ifft(fs, 2)

        f[:, j, :] .= real(fs)
    end
end

#=
function [ f ] = hf_lapenta2d(dt,Nvy,Vx,kx,f)

for j=1:Nvy
    fs=squeeze(f(:,j,:)); %fn
    fs=real(ifft((exp(-sqrt(-1)*Vx'.*kx*dt)).*fft(fs')))'; %f(n+1)=exp(-ikvx dt)fn
    f(:,j,:)=fs; %f(n+1)
end

end

=#
