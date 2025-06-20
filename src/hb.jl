using .Threads

struct OperatorHB{T}

    mesh::Mesh1D2V{T}
    kx::Vector{T}
    kvx::Vector{T}
    kvy::Vector{T}
    ft::Array{Complex{T}, 3}

    function OperatorHB(mesh::Mesh1D2V{T}) where T

        kx = collect(2π / (mesh.xmax - mesh.xmin) * fftfreq(mesh.nx, mesh.nx))
        kvx = collect(2π / (mesh.vxmax - mesh.vxmin) * fftfreq(mesh.nvx, mesh.nvx))
        kvy = collect(2π / (mesh.vymax - mesh.vymin) * fftfreq(mesh.nvy, mesh.nvy))

        ft = zeros(Complex{T}, (mesh.nvx, mesh.nvy, mesh.nx))

        new{T}(mesh, kx, kvx, kvy, ft)

    end

end
"""
solve the subsystem 

```math
df/dt+B Jv.df/dv=0 
dEy/dt=-dB/dx
dB/dt=drx/dt=dry/dt=0
```

first system equation is solved with semi-Lagrangian + exact splitting 

"""
function solve!(f, Ey, B, hb::OperatorHB, dt)

    nvx, nvy, nx = size(f)

    hdt = 0.5dt

    hb.ft .= fft(f, 1)
    @threads for l in 1:nx
        for j in 1:nvy, i in 1:nvx
           hb.ft[i, j, l] *= exp(-1im * hb.kvx[i] * hb.mesh.vy[j] * tan(B[l] * hdt))
        end
    end
    hb.ft .= ifft(hb.ft, 1)

    hb.ft .= fft(hb.ft, 2)
    @threads for l in 1:nx
        for i in 1:nvx, j in 1:nvy
            hb.ft[i, j, l] *= exp(1im * hb.kvy[j] * hb.mesh.vx[i] * sin(B[l] * dt))
        end
    end
    hb.ft .= ifft(hb.ft, 2)

    hb.ft .= fft(hb.ft, 1)
    @threads for l in 1:nx
        for  j in 1:nvy, i in 1:nvx
            hb.ft[i, j, l] *= exp(-1im * hb.kvx[i] * hb.mesh.vy[j] * tan(B[l] * hdt))
        end
    end
    hb.ft .= ifft(hb.ft, 1)


    return Ey .= Ey .- dt .* real(ifft(1im .* hb.kx .* fft(B)))

end

#=
function [ fn,Ey ] = HB(fn,Ey,B,vx,vy,kk,kkvx,kkvy,Nvx,Nvy,dvx,dvy,Nx,dt,d,meth)

%solve the subsystem 
%df/dt+B Jv.df/dv=0 (with semi-Lagrangian + exact splitting) 
%dEy/dt=-dB/dx
%dB/dt=drx/dt=dry/dt=0

if (meth==0)
    fn=fft(fn,[],1);
    for l=1:Nx
      for j=1:Nvy
        for i=1:Nvx
            fn(i,j,l)=exp(-1i*kkvx(i)*vy(j)*tan(B(l)*dt/2)).*fn(i,j,l);
        end
      end
    end
    fn=ifft(fn,[],1);    

    fn=fft(fn,[],2);
    for l=1:Nx
     for i=1:Nvx
         for j=1:Nvy
             fn(i,j,l)=exp(1i*kkvy(j)*vx(i)*sin(B(l)*dt)).*fn(i,j,l);
         end
      end
    end
    fn=ifft(fn,[],2); 

    fn=fft(fn,[],1);
    for l=1:Nx
      for j=1:Nvy
        for i=1:Nvx
            fn(i,j,l)=exp(-1i*kkvx(i)*vy(j)*tan(B(l)*dt/2)).*fn(i,j,l);
        end
      end
    end
    fn=ifft(fn,[],1);  

end

if (meth==1)

    for j=1:Nvy
        for i=1:Nx
            ft=fn(:,j,i)';
            dep=vy(j)*(tan(B(i)*dt/2))/(dt/2);
            u1=semilag(ft,Nvx,dep,dt/2,dvx,d);
            fn(:,j,i)=u1;
        end
    end

    for j=1:Nvx
        for i=1:Nx
            ft=fn(j,:,i);
            dep=-vx(j)*(sin(B(i)*dt))/dt;
            u1=semilag(ft,Nvy,dep,dt,dvy,d);
            fn(j,:,i)=u1;
        end
    end

    for j=1:Nvy
        for i=1:Nx
            ft=fn(:,j,i)';
            dep=vy(j)*(tan(B(i)*dt/2))/(dt/2);
            u1=semilag(ft,Nvx,dep,dt/2,dvx,d);
            fn(:,j,i)=u1;
        end
    end

end

   amperey=real(ifft(sqrt(-1)*kk.*fft(B)));
   Ey=Ey-dt*amperey;
end

=#

function HB(
        fn::Array{ComplexF64, 3},
        Ey::Vector{Float64},
        B::Vector{Float64},
        vx::Vector{Float64},
        vy::Vector{Float64},
        kk::Vector{Float64},
        kkvx::Vector{Float64},
        kkvy::Vector{Float64},
        Nvx::Int,
        Nvy::Int,
        dvx::Float64,
        dvy::Float64,
        Nx::Int,
        dt::Float64,
        d::Int,
        meth::Int,
    )
    if meth == 0
        # FFT-based method (Strang splitting)
        fn = fft(fn, 1)  # FFT along first dimension

        # First tan(B*dt/2) term
        @inbounds Threads.@threads for l in 1:Nx
            Bdt = B[l] * dt
            phase = -im * tan(Bdt / 2)
            for j in 1:Nvy
                vj = vy[j]
                for i in 1:Nvx
                    fn[i, j, l] *= exp(phase * kkvx[i] * vj)
                end
            end
        end

        fn = ifft(fn, 1)
        fn = fft(fn, 2)  # FFT along second dimension

        # sin(B*dt) term
        @inbounds Threads.@threads for l in 1:Nx
            Bdt = B[l] * dt
            phase = im * sin(Bdt)
            for i in 1:Nvx
                vi = vx[i]
                for j in 1:Nvy
                    fn[i, j, l] *= exp(phase * kkvy[j] * vi)
                end
            end
        end

        fn = ifft(fn, 2)
        fn = fft(fn, 1)

        # Second tan(B*dt/2) term
        @inbounds Threads.@threads for l in 1:Nx
            Bdt = B[l] * dt
            phase = -im * tan(Bdt / 2)
            for j in 1:Nvy
                vj = vy[j]
                for i in 1:Nvx
                    fn[i, j, l] *= exp(phase * kkvx[i] * vj)
                end
            end
        end

        fn = ifft(fn, 1)

    else
        # Semi-Lagrangian method
        fn_temp = similar(fn)

        # First advection step
        @inbounds Threads.@threads for j in 1:Nvy
            vj = vy[j]
            for i in 1:Nx
                ft = @view fn[:, j, i]
                dep = vj * tan(B[i] * dt / 2) / (dt / 2)
                fn_temp[:, j, i] = semilag(ft, Nvx, dep, dt / 2, dvx, d)
            end
        end
        fn .= fn_temp

        # Second advection step
        @inbounds Threads.@threads for j in 1:Nvx
            vj = vx[j]
            for i in 1:Nx
                ft = @view fn[j, :, i]
                dep = -vj * sin(B[i] * dt) / dt
                fn_temp[j, :, i] = semilag(ft, Nvy, dep, dt, dvy, d)
            end
        end
        fn .= fn_temp

        # Third advection step
        @inbounds Threads.@threads for j in 1:Nvy
            vj = vy[j]
            for i in 1:Nx
                ft = @view fn[:, j, i]
                dep = vj * tan(B[i] * dt / 2) / (dt / 2)
                fn_temp[:, j, i] = semilag(ft, Nvx, dep, dt / 2, dvx, d)
            end
        end
        fn .= fn_temp
    end

    # Update Ey field
    B_hat = fft(B)
    amperey = real(ifft(im .* kk .* B_hat))
    Ey .-= dt .* amperey

    return fn, Ey
end
