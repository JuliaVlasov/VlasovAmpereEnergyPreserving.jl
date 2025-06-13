function hf!(
    f::Matrix{Float64},
    E::Vector{Float64},
    dt::Float64,
    dv::Float64,
    v::Vector{Float64},
    kx::Vector{Float64},
)
    rhoh = vec(sum(f; dims=1)) .* dv
    ft = fft(f, 2)
    ft .*= exp.(-1im * v .* kx' * dt)
    f .= real(ifft(ft, 2))
    rhon = vec(sum(f; dims=1)) .* dv
    Ef = fft(E)
    rhs = fft(rhon .- rhoh)

    nx = length(kx)

    for i in 2:nx
        Ef[i] = Ef[i] + rhs[i] / (1im * kx[i])
    end

    Ef[1] = 0

    E .= real(ifft(Ef))
end

# """
# solve the subsystem
# 
# df/dt + vx df/dx = 0
# dEx/dt = -Jx
# dEy/dt = -Jy
# dB/dt=drx/dt=dry/dt=0
# """
function hf!(f, Ex, Ey, dt, mesh::Mesh1D2V, kx)

    nvx, nvy, nx = size(f)

    dvx = mesh.dvx
    dvy = mesh.dvy

    rhoh = vec(sum(f; dims=[1, 2])) .* dvx .* dvy

    # Initialize arrays
    fnsvy = similar(f)
    fnp1vy = similar(f)

    fs = complex(zeros(nvx, nx))

    for j in axes(f, 2)
        fs .= f[:, j, :]
        fnsvy[:, j, :] .= mesh.vy[j] .* fs ./ mesh.vx
        fs .= fft(fs, 2)
        fs .*= exp.(-1im .* mesh.vx .* kx' * dt)
        fs .= ifft(fs, 2)
        f[:, j, :] .= real(fs)
        fnp1vy[:, j, :] .= mesh.vy[j] * real(fs) ./ mesh.vx
    end

    # Calculate updated charge densities
    rhohvy = vec(sum(fnsvy; dims=[1, 2])) .* dvx .* dvy
    rhon = vec(sum(f; dims=[1, 2])) .* dvx .* dvy
    rhonvy = vec(sum(fnp1vy; dims=[1, 2])) .* dvx .* dvy

    # Compute current and update electric fields
    k_mod = copy(kx)
    k_mod[1] = 1  # Avoid division by zero for DC component

    # Update Ex field
    rhs = (1 ./ (1im * k_mod)) .* fft(rhon .- rhoh)
    Ef = fft(Ex)
    Ef .+= rhs
    Ef[1] = 0  # Set DC component to zero
    Ex .= real(ifft(Ef))

    # Update Ey field
    rhs = (1 ./ (1im * k_mod)) .* fft(rhonvy .- rhohvy)
    Ef = fft(Ey)
    Ef .+= rhs
    Ef[1] = 0  # Set DC component to zero
    Ey .= real(ifft(Ef))
end

#=
function [ f, Ex, Ey ] = hf(dt,dvx,dvy,Nvx,Nvy,Nx,Vx,vy,kx,k,f,Ex,Ey)

%solve the subsystem 
%df/dt + vx df/dx = 0 
%dEx/dt = -Jx 
%dEy/dt = -Jy
%dB/dt=drx/dt=dry/dt=0

rhoh=squeeze(sum(sum(f))*dvx*dvy)';
for j=1:Nvy
    fs=squeeze(f(:,j,:)); %fn
    fnsvy(:,j,:)=vy(j)*fs./Vx;
    fs=real(ifft((exp(-sqrt(-1)*Vx'.*kx*dt)).*fft(fs')))'; %f(n+1)=exp(-ikvx dt)fn
    f(:,j,:)=fs; %f(n+1)
    fnp1vy(:,j,:)=vy(j)*fs./Vx;
end

rhohvy=squeeze(sum(sum(fnsvy))*dvx*dvy)';
rhon=squeeze(sum(sum(f))*dvx*dvy)';
rhonvy=squeeze(sum(sum(fnp1vy))*dvx*dvy)';

%compute the current exactly in time by hand to get Ex and Ey

k_mod=k;
k_mod(1)=1;

rhs=(1./(sqrt(-1)*k_mod)).*fft(rhon-rhoh);
Ef=fft(Ex);

Ef=Ef+rhs;
Ef(1,1)=0;
Ex=real(ifft(Ef));

rhs=(1./(sqrt(-1)*k_mod)).*(fft(rhonvy-rhohvy));
Ef=fft(Ey);
Ef=Ef+rhs;
Ef(1,1)=0;
Ey=real(ifft(Ef));

end
=#

export hf

"""
solve the subsystem

df/dt + vx df/dvx = 0
dEx/dt = -Jx
dEy/dt = -Jy
dB/dt=drx/dt=dry/dt=0

"""
function hf(dt, dvx, dvy, Nvx, Nvy, Nx, Vx, vy, kx, k, f, Ex, Ey)
    # Calculate initial charge densities
    rhoh = dropdims(sum(sum(f; dims=1); dims=2)) * dvx * dvy

    # Initialize arrays
    fnsvy = similar(f)
    fnp1vy = similar(f)

    # Loop over vy dimension
    for j in 1:Nvy
        fs = dropdims(f[:, j, :]; dims=2)  # Extract 2D slice
        fnsvy[:, j, :] = vy[j] * fs ./ Vx

        # Fourier transform and apply phase shift
        fs_hat = fft(fs', 1)  # FFT along x dimension
        fs_shifted = exp.(-im * Vx' .* kx * dt) .* fs_hat
        fs = real(ifft(fs_shifted, 1))'  # Inverse FFT

        f[:, j, :] = fs  # Update distribution function
        fnp1vy[:, j, :] = vy[j] * fs ./ Vx
    end

    # Calculate updated charge densities
    rhohvy = dropdims(sum(sum(fnsvy; dims=1); dims=2) * dvx * dvy)
    rhon = dropdims(sum(sum(f; dims=1); dims=2) * dvx * dvy)
    rhonvy = dropdims(sum(sum(fnp1vy; dims=1); dims=2) * dvx * dvy)

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
