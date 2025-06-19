"""
solve the subsystem 

    df/dt+B Jv.df/dv=0 (with semi-Lagrangian + exact splitting) 
    dEy/dt = -dB/dx
    dB/dt  = -dEy/dx 
    dEx/dt = 0

"""
function HB_lapenta2d!(f, Ex, Ey, B, kx, kvx, kvy, mesh::Mesh1D2V, dt)

    hdt = 0.5dt

    Eyf = fft(Ey)
    Bf = fft(B)
    Eynp1f = similar(Eyf)
    Bnp1f = similar(Bf)

    for i in eachindex(kx)
        c = 0.5dt * 1im * kx[i]

        AA = [1 c; c 1]
        Un = [Eyf[i] - c * Bf[i], Bf[i] - c * Eyf[i]]

        U = AA \ Un
        Eynp1f[i] = U[1]
        Bnp1f[i] = U[2]
    end

    Eynp1 = real(ifft(Eynp1f))
    Bnp1 = real(ifft(Bnp1f))

    Bdemi = 0.5 * (Bnp1 .+ B)

    ft = complex(f)
    ft .= fft(f, 1)
    for l in axes(ft, 3), j in axes(ft, 2), i in axes(ft, 1)
        ft[i, j, l] = exp(-1im * kvx[i] * mesh.vy[j] * tan(Bdemi[l] * hdt)) * ft[i, j, l]
    end
    ft .= ifft(ft, 1)

    ft .= fft(ft, 2)
    for l in axes(ft, 3), j in axes(ft, 2), i in axes(ft, 1)
        ft[i, j, l] = exp(1im * kvy[j] * mesh.vx[i] * sin(Bdemi[l] * dt)) * ft[i, j, l]
    end
    ft .= ifft(ft, 2)

    ft .= fft(ft, 1)
    for l in axes(ft, 3), j in axes(ft, 2), i in axes(ft, 1)
        ft[i, j, l] = exp(- 1im * kvx[i] * mesh.vy[j] * tan(Bdemi[l] * hdt)) * ft[i, j, l]
    end
    ft .= ifft(ft, 1)

    Ey .= Eynp1
    B .= Bnp1
    return f .= real(ft)
end

#=
function [ fn,Ex,Ey,B ] = HB_lapenta2d(fn,Ex,Ey,B,vx,vy,kk,kkvx,kkvy,Nvx,Nvy,dvx,dvy,Nx,dt,dx,d,meth)

%solve the subsystem 
%df/dt+B Jv.df/dv=0 (with semi-Lagrangian + exact splitting) 
%dEy/dt = -dB/dx
%dB/dt  = -dEy/dx 
%dEx/dt = 0

Eyf=fft(Ey);
Bf=fft(B);
for i=1:Nx 
    AA=[1, 0.5*dt*sqrt(-1)*kk(i);0.5*dt*sqrt(-1)*kk(i),1]; 
    Un=[Eyf(i)-0.5*dt*sqrt(-1)*kk(i)*Bf(i);Bf(i)-0.5*dt*sqrt(-1)*kk(i)*Eyf(i)];
    U=inv(AA)*Un;
    Eynp1f(i)=U(1);Bnp1f(i)=U(2);
end

Eynp1=real(ifft(Eynp1f));
Bnp1=real(ifft(Bnp1f));

Bdemi=0.5*(Bnp1+B);
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

%update Ey and B 
Ey=Eynp1;
B=Bnp1;

end

=#

using FFTW
using LinearAlgebra

function HB_lapenta2d(
        fn, Ex, Ey, B, vx, vy, kk, kkvx, kkvy, Nvx, Nvy, dvx, dvy, Nx, dt, dx, d, meth
    )

    # solve the subsystem
    # df/dt + B Jv.df/dv = 0 (with semi-Lagrangian + exact splitting)
    # dEy/dt = -dB/dx
    # dB/dt  = -dEy/dx
    # dEx/dt = 0

    Eyf = fft(Ey)
    Bf = fft(B)
    Eynp1f = similar(Eyf)
    Bnp1f = similar(Bf)

    for i in 1:Nx
        AA = [1 0.5 * dt * im * kk[i]; [0.5 * dt * im * kk[i] 1]]
        Un = [Eyf[i] - 0.5 * dt * im * kk[i] * Bf[i]; Bf[i] - 0.5 * dt * im * kk[i] * Eyf[i]]
        U = AA \ Un
        Eynp1f[i] = U[1]
        Bnp1f[i] = U[2]
    end

    Eynp1 = real(ifft(Eynp1f))
    Bnp1 = real(ifft(Bnp1f))

    Bdemi = 0.5 * (Bnp1 + B)

    if meth == 0
        fn = fft(fn, 1)  # FFT along first dimension
        for l in 1:Nx
            for j in 1:Nvy
                for i in 1:Nvx
                    fn[i, j, l] = exp(-im * kkvx[i] * vy[j] * tan(Bdemi[l] * dt / 2)) * fn[i, j, l]
                end
            end
        end
        fn = ifft(fn, 1)

        fn = fft(fn, 2)  # FFT along second dimension
        for l in 1:Nx
            for i in 1:Nvx
                for j in 1:Nvy
                    fn[i, j, l] = exp(im * kkvy[j] * vx[i] * sin(Bdemi[l] * dt)) * fn[i, j, l]
                end
            end
        end
        fn = ifft(fn, 2)

        fn = fft(fn, 1)  # FFT along first dimension
        for l in 1:Nx
            for j in 1:Nvy
                for i in 1:Nvx
                    fn[i, j, l] = exp(-im * kkvx[i] * vy[j] * tan(Bdemi[l] * dt / 2)) * fn[i, j, l]
                end
            end
        end
        fn = ifft(fn, 1)
    end

    if meth == 1
        for j in 1:Nvy
            for i in 1:Nx
                ft = fn[:, j, i]'
                dep = vy[j] * (tan(Bdemi[i] * dt / 2)) / (dt / 2)
                u1 = semilag(ft, Nvx, dep, dt / 2, dvx, d)
                fn[:, j, i] = u1
            end
        end

        for j in 1:Nvx
            for i in 1:Nx
                ft = fn[j, :, i]
                dep = -vx[j] * (sin(Bdemi[i] * dt)) / dt
                u1 = semilag(ft, Nvy, dep, dt, dvy, d)
                fn[j, :, i] = u1
            end
        end

        for j in 1:Nvy
            for i in 1:Nx
                ft = fn[:, j, i]'
                dep = vy[j] * (tan(Bdemi[i] * dt / 2)) / (dt / 2)
                u1 = semilag(ft, Nvx, dep, dt / 2, dvx, d)
                fn[:, j, i] = u1
            end
        end
    end

    # update Ey and B
    Ey = Eynp1
    B = Bnp1

    return fn, Ex, Ey, B
end
