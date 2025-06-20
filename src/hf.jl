struct OperatorHF{T}

    mesh::AbstractMesh
    kx::Vector{T}
    kmod::Vector{T}
    ft::Matrix{Complex{T}}
    rhoh::Vector{T}
    rhohvy::Vector{T}
    rhon::Vector{T}
    rhonvy::Vector{T}
    rhs::Vector{Complex{T}}
    Ef::Vector{Complex{T}}

    function OperatorHF(mesh::Mesh1D2V{T}) where {T}

        kx = collect(2π / (mesh.xmax - mesh.xmin) .* fftfreq(mesh.nx, mesh.nx))
        kmod = copy(kx)
        kmod[1] = 1
        ft = zeros(Complex{T}, (mesh.nvx, mesh.nx))
        rhoh = zeros(mesh.nx)
        rhohvy = zeros(mesh.nx)
        rhon = zeros(mesh.nx)
        rhonvy = zeros(mesh.nx)
        rhs = zeros(Complex{T}, mesh.nx)
        Ef = zeros(Complex{T}, mesh.nx)

        return new{T}(mesh, kx, kmod, ft, rhoh, rhohvy, rhon, rhonvy, rhs, Ef)

    end

end

"""
solve the subsystem

```math
df/dt + vx df/dx = 0
dEx/dt = -Jx
dEy/dt = -Jy
dB/dt = drx/dt = dry/dt = 0
```
"""
function solve!(f, Ex, Ey, hf::OperatorHF, dt)

    nvx, nvy, nx = size(f)

    dvx = hf.mesh.dvx
    dvy = hf.mesh.dvy

    fnsvy = similar(f)
    fnp1vy = similar(f)

    rhoh = vec(sum(f; dims = [1, 2])) .* dvx .* dvy

    for j in axes(f, 2)
        hf.ft .= f[:, j, :]
        fnsvy[:, j, :] .= hf.mesh.vy[j] .* hf.ft ./ hf.mesh.vx
        hf.ft .= fft(hf.ft, 2)
        hf.ft .*= exp.(-1im .* hf.mesh.vx .* hf.kx' * dt)
        hf.ft .= ifft(hf.ft, 2)
        f[:, j, :] .= real(hf.ft)
        fnp1vy[:, j, :] .= hf.mesh.vy[j] * real(hf.ft) ./ hf.mesh.vx
    end

    rhon = vec(sum(f; dims = [1, 2])) .* dvx .* dvy
    rhohvy = vec(sum(fnsvy; dims = [1, 2])) .* dvx .* dvy
    rhonvy = vec(sum(fnp1vy; dims = [1, 2])) .* dvx .* dvy


    rhs = (1 ./ (1im * hf.kmod)) .* fft(rhon .- rhoh)
    Ef = fft(Ex)
    Ef .+= rhs
    Ef[1] = 0
    Ex .= real(ifft(Ef))

    rhs = (1 ./ (1im * hf.kmod)) .* fft(rhonvy .- rhohvy)
    Ef = fft(Ey)
    Ef .+= rhs
    Ef[1] = 0
    return Ey .= real(ifft(Ef))
end

function hf!(
        f::Matrix{Float64},
        E::Vector{Float64},
        dt::Float64,
        dv::Float64,
        v::Vector{Float64},
        kx::Vector{Float64},
    )
    rhoh = vec(sum(f; dims = 1)) .* dv
    ft = fft(f, 2)
    ft .*= exp.(-1im * v .* kx' * dt)
    f .= real(ifft(ft, 2))
    rhon = vec(sum(f; dims = 1)) .* dv
    Ef = fft(E)
    rhs = fft(rhon .- rhoh)

    nx = length(kx)

    for i in 2:nx
        Ef[i] = Ef[i] + rhs[i] / (1im * kx[i])
    end

    Ef[1] = 0

    return E .= real(ifft(Ef))
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
