"""
solve the subsystem 
df/dt + B Jv . df/dv = 0
dB/dt = -ry d(gy)/dx 
dEy/dt=-dB/dx
dry/dt=-0.5*int (gy dB/dx)) dx = 0.5*int (d(gy)/dx B) dx 
dEx/dt=0
"""
function HB2!(f, Ey, B, gx, gy, rx, ry, kx, kvx, kvy, mesh, dt)

    nvx, nvy, nx = size(f)

    dx = mesh.dx
    hdt = 0.5dt

    dxgy = zeros(nx)

    for i in 2:(nx - 1)
        dxgy[i]=(gy[i + 1]-gy[i - 1]) / (2dx)
    end
    dxgy[1]=(gy[2]-gy[nx])/(2dx)
    dxgy[nx]=(gy[1]-gy[nx - 1])/(2dx)

    denom=1+(dt*dt/8.0)*sum(dxgy .* dxgy)*dx
    rynp1=(ry+0.5*dt*sum(B .* dxgy)*dx-(dt*dt/8.0)*ry*sum(dxgy .* dxgy)*dx)/denom;
    rydemi=ry/denom+0.25*dt*sum(B .* dxgy)*dx/denom;

    Bnp1=B-dt*rydemi*dxgy
    Bdemi=B-0.5*dt*rydemi*dxgy

    dxBdemi = zeros(nx)
    for i in 2:(nx - 1)
        dxBdemi[i]=(Bdemi[i + 1]-Bdemi[i - 1])/(2dx)
    end


    dxBdemi[1]=(Bdemi[2]-Bdemi[nx])/(2dx)
    dxBdemi[nx]=(Bdemi[1]-Bdemi[nx - 1])/(2dx)

    Eynp1=Ey-dt*dxBdemi

    ft = complex(f)

    ft .= fft(f, 1)
    for l in 1:nx, j in 1:nvy, i in 1:nvx
        ft[i, j, l] *= exp(-1im*kvx[i]*mesh.vy[j]*tan(Bdemi[l]*hdt))
    end
    ft .= ifft(ft, 1)

    ft .= fft(ft, 2)
    for l in 1:nx, i in 1:nvx, j in 1:nvy
        ft[i, j, l] *= exp(1im*kvy[j]*mesh.vx[i]*sin(Bdemi[l]*dt))
    end
    ft .= ifft(ft, 2)

    ft .= fft(ft, 1)
    for l in 1:nx, j in 1:nvy, i in 1:nvx
        ft[i, j, l] *= exp(-1im*kvx[i]*mesh.vy[j]*tan(Bdemi[l]*hdt))
    end
    ft .= ifft(ft, 1)

    f .= real(ft)
    Ey .= Eynp1
    B .= Bnp1

    return rx, rynp1

end
#=
function [ fn,Ey,ry,B ] = HB2(fn,vx,vy,Ey,ry,gy,B,Nvx,Nvy,kk,kkvx,kkvy,dvx,dvy,Nx,dx,dt,d,meth)

%solve the subsystem 
%df/dt + B Jv . df/dv = 0
%dB/dt = -ry d(gy)/dx 
%dEy/dt=-dB/dx
%dry/dt=-0.5*int (gy dB/dx)) dx = 0.5*int (d(gy)/dx B) dx 
%dEx/dt=0

%update ry
%dxgy=real(ifft(sqrt(-1)*kk.*fft(gy)));
for i=2:Nx-1
    dxgy(i)=(gy(i+1)-gy(i-1))/(2.0*dx);
end
dxgy(1)=(gy(2)-gy(Nx))/(2.0*dx);
dxgy(Nx)=(gy(1)-gy(Nx-1))/(2.0*dx);
denom=1+(dt*dt/8.0)*sum(dxgy.*dxgy)*dx;
rynp1=(ry+0.5*dt*sum(B.*dxgy)*dx-(dt*dt/8.0)*ry*sum(dxgy.*dxgy)*dx)/denom;
rydemi=ry/denom+0.25*dt*sum(B.*dxgy)*dx/denom;
%disp('verif')
%disp(0.5*(rynp1+ry)-rydemi)

%update B
Bnp1=B-dt*rydemi*dxgy;
Bdemi=B-0.5*dt*rydemi*dxgy;
%update Ey
%dxBdemi=real(ifft(sqrt(-1)*kk.*fft(Bdemi)));
for i=2:Nx-1
    dxBdemi(i)=(Bdemi(i+1)-Bdemi(i-1))/(2.0*dx);
end
dxBdemi(1)=(Bdemi(2)-Bdemi(Nx))/(2.0*dx);
dxBdemi(Nx)=(Bdemi(1)-Bdemi(Nx-1))/(2.0*dx);

Eynp1=Ey-dt*dxBdemi;

%update f
if (meth==0)
    fn=fft(fn,[],1);
    for l=1:Nx
      for j=1:Nvy
        for i=1:Nvx
            fn(i,j,l)=exp(-1i*kkvx(i)*vy(j)*tan(Bdemi(l)*dt/2)).*fn(i,j,l);
        end
      end
    end
    fn=ifft(fn,[],1);    

    fn=fft(fn,[],2);
    for l=1:Nx
     for i=1:Nvx
         for j=1:Nvy
             fn(i,j,l)=exp(1i*kkvy(j)*vx(i)*sin(Bdemi(l)*dt)).*fn(i,j,l);
         end
      end
    end
    fn=ifft(fn,[],2); 

    fn=fft(fn,[],1);
    for l=1:Nx
      for j=1:Nvy
        for i=1:Nvx
            fn(i,j,l)=exp(-1i*kkvx(i)*vy(j)*tan(Bdemi(l)*dt/2)).*fn(i,j,l);
        end
      end
    end
    fn=ifft(fn,[],1);  

end

if (meth==1)

    for j=1:Nvy
        for i=1:Nx
            ft=fn(:,j,i)';
            dep=vy(j)*(tan(Bdemi(i)*dt/2))/(dt/2);
            u1=semilag(ft,Nvx,dep,dt/2,dvx,d);
            fn(:,j,i)=u1;
        end
    end

    for j=1:Nvx
        for i=1:Nx
            ft=fn(j,:,i);
            dep=-vx(j)*(sin(Bdemi(i)*dt))/dt;
            u1=semilag(ft,Nvy,dep,dt,dvy,d);
            fn(j,:,i)=u1;
        end
    end

    for j=1:Nvy
        for i=1:Nx
            ft=fn(:,j,i)';
            dep=vy(j)*(tan(Bdemi(i)*dt/2))/(dt/2);
            u1=semilag(ft,Nvx,dep,dt/2,dvx,d);
            fn(:,j,i)=u1;
        end
    end

end

Ey=Eynp1;
ry=rynp1;
B=Bnp1;

end

=#
