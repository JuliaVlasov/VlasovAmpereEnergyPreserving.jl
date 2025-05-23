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

