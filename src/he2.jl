#=
function [ fn,rx,ry, gx, gy ] = HE2(fn,vx,vy,rx,gx,Ex,ry,gy,Ey,Nvx,Nvy,dvx,dvy,Nx,dx,x,dt,d)

%solve the subsystem 
%df/dt + rxgx df/dvx + rygydf/dvy = 0
%drx/dt=-0.5*int (gx Jx) dx
%dry/dt=-0.5*int (gy Jy) dx
%dEx/dt=dEy/dt=dB/dt =0


for i=1:Nx
    Jx(i)=0.;Jy(i)=0.;rho(i)=0.;
    for j=1:Nvx
        for k=1:Nvy
            Jx(i) =Jx(i) +vx(j)*fn(j,k,i);
            Jy(i) =Jy(i) +vy(k)*fn(j,k,i);
            rho(i)=rho(i)+      fn(j,k,i);
        end
    end
end
Jx(1:Nx)=Jx(1:Nx)*dvx*dvy;Jy(1:Nx)=Jy(1:Nx)*dvx*dvy;rho(1:Nx)=rho(1:Nx)*dvx*dvy;

[gx,gy]=updateg(rx,Ex,ry,Ey,x,Nx);

%update rx
denom=1+(dt*dt/8.0)*sum(rho.*gx.^2)*dx;
rxnp1=(rx-0.5*dt*sum(gx.*Jx)*dx-(dt*dt/8.0)*rx*sum(rho.*gx.^2)*dx)/denom;
rxdemi=rx/denom-(0.25*dt/denom)*sum(gx.*Jx)*dx;
Exnew=rxdemi*gx;

%update ry 
denom=1+(dt*dt/8.0)*sum(rho.*gy.^2)*dx;
rynp1=(ry-0.5*dt*sum(Jy.*gy)*dx-(dt*dt/8.0)*ry*sum(rho.*gy.^2)*dx)/denom;
rydemi=ry/denom-(0.25*dt/denom)*sum(gy.*Jy)*dx;
Eynew=rydemi*gy;

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
