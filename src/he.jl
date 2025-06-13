"""
solve the subsystem 
df/dt + rxgx df/dvx + rygydf/dvy = 0
dB/dt = -d(ry gy)/dx 
drx/dt=-0.5*int (gx Jx) dx
dry/dt=-0.5*int (gy (Jy+dB/dx)) dx
dEx/dt=dEy/dt=0
"""
function HE!(f, rx, ry, B, vx, vy, gx, gy, kx, dx, dt)
    nvx, nvy, nx = size(f)

    dvx = vx[2]-vx[1]
    dvy = vy[2]-vy[1]
    Jx = zeros(nx)
    Jy = zeros(nx)
    rho = zeros(nx)

    Jx = vec(sum(vx .* f; dims=[1, 2])) .* dvx .* dvy
    Jy = vec(sum(vy' .* f; dims=[1, 2])) .* dvx .* dvy
    rho = vec(sum(f; dims=[1, 2])) .* dvx .* dvy

    Jxnpdemi = Jx + 0.25dt * rx * gx .* rho

    denom=1+(dt*dt/8)*sum(rho .* gx .^ 2)*dx

    rxnp1=(rx-0.5dt*sum(gx .* Jx)*dx-(dt*dt/8)*rx*sum(rho .* gx .^ 2)*dx)/denom

    Exnew=0.5*(rxnp1+rx)*gx;

    Jynpdemi = Jy + 0.25dt * ry * gy .* rho
    dxgy = real(ifft(1im * kx .* fft(gy)))
    dx2gy = real(ifft(-kx .* kx .* fft(gy)))
    dxB = real(ifft(1im * kx .* fft(B)))
    denom = 1+(dt*dt/8)*sum(rho .* gy .^ 2)*dx+(dt*dt/8)*sum(dxgy .* dxgy)*dx

    rynp1 = ry - 0.5 * dt * sum(Jy .* gy) * dx
    rynp1 += 0.5dt * sum(B .* dxgy) * dx
    rynp1 -= (dt*dt/8)*ry*sum(rho .* gy .^ 2)*dx
    rynp1 -= (dt*dt/8)*ry*sum(dxgy .* dxgy)*dx

    rynp1 /= denom

    Eynew = 0.5 * (rynp1+ry) * gy

    faraday=real(ifft(1im*kx .* fft(Eynew)))

    B .= B .- dt * faraday

    fvx = zeros(nvx)
    for j in axes(f, 2), i in eachindex(Exnew)
        fvx = f[:, j, i]
        f[:, j, i] .= semilag_d2(fvx, Exnew[i], dt, dvx)
    end

    fvy = zeros(nvy)
    for j in axes(f, 1), i in eachindex(Exnew)
        fvy = f[j, :, i]
        f[j, :, i] .= semilag_d2(fvy, Eynew[i], dt, dvy)
    end

    rxnp1, rynp1
end

#=
function [ fn,rx,ry,B ] = HE(fn,vx,vy,Vx,Vy,rx,gx,ry,gy,B,Nvx,Nvy,kk,kv,dvx,dvy,Nx,dx,dt,d)

%solve the subsystem 
%df/dt + rxgx df/dvx + rygydf/dvy = 0
%dB/dt = -d(ry gy)/dx 
%drx/dt=-0.5*int (gx Jx) dx
%dry/dt=-0.5*int (gy (Jy+dB/dx)) dx
%dEx/dt=dEy/dt=0

for i=1:Nx
    Jx(i)=0.;Jy(i)=0.;rho(i)=0;
    for j=1:Nvx
        for k=1:Nvy
            Jx(i)=Jx(i)+vx(j)*fn(j,k,i);
            Jy(i)=Jy(i)+vy(k)*fn(j,k,i);
            rho(i)=rho(i)+fn(j,k,i);
        end
    end
end
Jx=Jx*dvx*dvy;Jy=Jy*dvx*dvy;rho=rho*dvx*dvy;

Jxnpdemi=Jx+0.25*dt*rx*gx.*rho;
denom=1+(dt*dt/8)*sum(rho.*gx.^2)*dx;
rxnp1=(rx-0.5*dt*sum(gx.*Jx)*dx-(dt*dt/8)*rx*sum(rho.*gx.^2)*dx)/denom;
Exnew=0.5*(rxnp1+rx)*gx;

Jynpdemi=Jy+0.25*dt*ry*gy.*rho;
dxgy=real(ifft(sqrt(-1)*kk.*fft(gy)));
dx2gy=real(ifft(-kk.*kk.*fft(gy)));
dxB=real(ifft(sqrt(-1)*kk.*fft(B)));
denom=1+(dt*dt/8)*sum(rho.*gy.^2)*dx+(dt*dt/8)*sum(dxgy.*dxgy)*dx;
rynp1=(ry-0.5*dt*sum(Jy.*gy)*dx+0.5*dt*sum(B.*dxgy)*dx...
    -(dt*dt/8)*ry*sum(rho.*gy.^2)*dx-(dt*dt/8)*ry*sum(dxgy.*dxgy)*dx)/denom;

Eynew=0.5*(rynp1+ry)*gy;

faraday=real(ifft(sqrt(-1)*kk.*fft(Eynew)));

B=B-dt*faraday;

for j=1:Nvy
    for i=1:Nx
        ft=fn(:,j,i)';
        u1=semilag(ft,Nvx,Exnew(i),dt,dvx,d);
        fn(:,j,i)=u1;
    end
end

for j=1:Nvx
    for i=1:Nx
        ft=fn(j,:,i);
        u1=semilag(ft,Nvy,Eynew(i),dt,dvy,d);
        fn(j,:,i)=u1;
    end
end

rx=rxnp1;
ry=rynp1;

end
=#

function HE(
    fn::Array{Float64,3},
    vx::Vector{Float64},
    vy::Vector{Float64},
    Vx::Matrix{Float64},
    Vy::Matrix{Float64},
    rx::Float64,
    gx::Vector{Float64},
    ry::Float64,
    gy::Vector{Float64},
    B::Vector{Float64},
    Nvx::Int,
    Nvy::Int,
    kk::Vector{Float64},
    kv::Matrix{Float64},
    dvx::Float64,
    dvy::Float64,
    Nx::Int,
    dx::Float64,
    dt::Float64,
    d::Int,
)

    # Pre-allocate arrays
    Jx = zeros(Nx)
    Jy = zeros(Nx)
    rho = zeros(Nx)

    # Compute moments (optimized with views and reduced allocations)
    @inbounds for i in 1:Nx
        fn_slice = view(fn,:,:,i)
        Jx[i] = sum(vx .* sum(fn_slice, dims=2)) * dvx * dvy
        Jy[i] = sum(vy' .* sum(fn_slice, dims=1)) * dvx * dvy
        rho[i] = sum(fn_slice) * dvx * dvy
    end

    # Update rx and Ex
    Jxnpdemi = Jx .+ 0.25 .* dt .* rx .* gx .* rho
    denom = 1 + (dt^2 / 8) * sum(rho .* gx .^ 2) * dx
    rxnp1 =
        (rx - 0.5 * dt * sum(gx .* Jx) * dx - (dt^2 / 8) * rx * sum(rho .* gx .^ 2) * dx) /
        denom
    Exnew = 0.5 .* (rxnp1 .+ rx) .* gx

    # Update ry and Ey
    Jynpdemi = Jy .+ 0.25 .* dt .* ry .* gy .* rho
    gy_hat = fft(gy)
    dxgy = real(ifft(im .* kk .* gy_hat))
    dx2gy = real(ifft(-kk .^ 2 .* gy_hat))
    dxB = real(ifft(im .* kk .* fft(B)))

    denom = 1 + (dt^2 / 8) * sum(rho .* gy .^ 2) * dx + (dt^2 / 8) * sum(dxgy .^ 2) * dx
    rynp1 =
        (
            ry - 0.5 * dt * sum(Jy .* gy) * dx + 0.5 * dt * sum(B .* dxgy) * dx -
            (dt^2 / 8) * ry * sum(rho .* gy .^ 2) * dx -
            (dt^2 / 8) * ry * sum(dxgy .^ 2) * dx
        ) / denom
    Eynew = 0.5 .* (rynp1 .+ ry) .* gy

    # Update B field
    faraday = real(ifft(im .* kk .* fft(Eynew)))
    B .-= dt .* faraday

    # Semi-Lagrangian steps (optimized with threading)
    fn_temp = similar(fn)

    # x-direction
    @inbounds Threads.@threads for j in 1:Nvy
        for i in 1:Nx
            ft = view(fn, :, j, i)
            fn_temp[:, j, i] = semilag(ft, Nvx, Exnew[i], dt, dvx, d)
        end
    end
    fn .= fn_temp

    # y-direction
    @inbounds Threads.@threads for j in 1:Nvx
        for i in 1:Nx
            ft = view(fn, j, :, i)
            fn_temp[j, :, i] = semilag(ft, Nvy, Eynew[i], dt, dvy, d)
        end
    end
    fn .= fn_temp

    return fn, rxnp1, rynp1, B
end
