"""
solve the subsystem 

math```
    df/dt + Ex df/dvx + Ey df/dvy = 0
    dEx/dt=-Jx
    dEy/dt=-Jy
    dB/dt=0
```

"""
function HE_lapenta2d!(f, Ex, Ey, mesh::Mesh1D2V, dt)

    nvx, nvy, nx = size(f)

    dvx = mesh.dvx
    dvy = mesh.dvy

    Jx = zeros(nx)
    Jy = zeros(nx)
    rho = zeros(nx)

    for ix in axes(f, 3), ivy in axes(f, 2), ivx in axes(f, 1)
        Jx[ix] += mesh.vx[ivx] * f[ivx, ivy, ix]
        Jy[ix] += mesh.vy[ivy] * f[ivx, ivy, ix]
        rho[ix] += f[ivx, ivy, ix]
    end

    Jx .*= dvx * dvy
    Jy .*= dvx * dvy
    rho .*= dvx * dvy

    dt2_rho4 = (dt^2) .* rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2

    Exnew = (1 ./ (1 .+ dt2_rho4)) .* Ex .- (dt ./ (2 .+ dt2_rho2)) .* Jx
    Eynew = (1 ./ (1 .+ dt2_rho4)) .* Ey .- (dt ./ (2 .+ dt2_rho2)) .* Jy

    fvx = zeros(nvx)
    for j in axes(f, 2), i in eachindex(Exnew)
        fvx .= f[:, j, i]
        fvx .= semilag_d2(fvx, Exnew[i], dt, dvx)
        f[:, j, i] .= fvx
    end

    fvy = zeros(nvy)
    for k in axes(f, 1), i in eachindex(Eynew)
        fvy .= f[k, :, i]
        fvy .= semilag_d2(fvy, Eynew[i], dt, dvy)
        f[k, :, i] .= fvy
    end

    Ex .= 2 .* Exnew .- Ex
    return Ey .= 2 .* Eynew .- Ey
end
#=
function [ fn,Ex,Ey ] = HE_lapenta2d(fn,vx,vy,Ex,Ey,Nvx,Nvy,dvx,dvy,Nx,dx,dt,d)

%solve the subsystem 
%df/dt + Ex df/dvx + Ey df/dvy = 0
%dEx/dt=-Jx
%dEy/dt=-Jy
%dB/dt=0

for i=1:Nx
    Jx(i)=0.;Jy(i)=0.;rho(i)=0;
    for j=1:Nvx-1
        for k=1:Nvy-1
            Jx(i)=Jx(i)+vx(j)*fn(j,k,i);
            Jy(i)=Jy(i)+vy(k)*fn(j,k,i);
            rho(i)=rho(i)+fn(j,k,i);
        end
    end
end
Jx=Jx*dvx*dvy;Jy=Jy*dvx*dvy;rho=rho*dvx*dvy;

Exnew=(1./(1+dt*dt*0.25*rho)).*Ex-0.5*(dt./(1+dt*dt*0.25*rho)).*Jx;
Eynew=(1./(1+dt*dt*0.25*rho)).*Ey-0.5*(dt./(1+dt*dt*0.25*rho)).*Jy;

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

Ex=2*Exnew-Ex;
Ey=2*Eynew-Ey;

end

=#
