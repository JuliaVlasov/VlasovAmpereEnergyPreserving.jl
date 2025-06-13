function hf_lapenta2d!(f, dt, vx, kx)

    for j in axes(f, 2)

        fs = fft(f[:, j, :], 1)
        fs .*= exp.(-1im * vx .* kx' * dt)
        fs .= ifft(fs, 1)

        f[:, j, :] .= real(fs)
    end

end

#=
function [ f ] = hf_lapenta2d(dt,Nvy,Vx,kx,f)

%solve the subsystem 
%df/dt + vx df/dvx = 0 
%dEx/dt = 0
%dEy/dt = 0
%dB/dt  = 0



for j=1:Nvy
    fs=squeeze(f(:,j,:)); %fn
    fs=real(ifft((exp(-sqrt(-1)*Vx'.*kx*dt)).*fft(fs')))'; %f(n+1)=exp(-ikvx dt)fn
    f(:,j,:)=fs; %f(n+1)
end

end

=#
