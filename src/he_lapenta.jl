#=
function [ fn1,Enp1 ] = HE_lapenta(fn,V,E,Nv,kv,dv,Nx,dx,dt,d)

    J=sum(V.*fn)*dv; 
    rho=sum(fn)*dv;

    Enew=(1./(1+(dt*dt*rho/4))).*E-(dt./(2+(dt*dt*rho/2))).*J;
    Jnp1=J+dt*Enew.*rho;
    Jnpdemi=J+0.5*dt*Enew.*rho;
    Enp1=E-dt*Jnpdemi;

    for i=1:Nx
        ft=fn(:,i)';
        u1=semilag(ft,Nv,Enew(i),dt,dv,d);
        fn1(:,i)=u1;
    end

end
=#

export HE_lapenta!

function HE_lapenta!(
        f::Matrix{T},
        E::Vector{T},
        v::Vector{T},
        nv::Int,
        kv::Vector{T},
        dv::T,
        nx::Int,
        dx::T,
        dt::T,
    ) where {T <: AbstractFloat}

    # Pre-allocate outputs
    fn1 = similar(f)

    J = vec(sum(v .* f; dims = 1)) .* dv
    rho = vec(sum(f; dims = 1)) .* dv

    dt2_rho4 = (dt^2) .* rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2

    Enew = E ./ (1 .+ dt2_rho4) .- (dt * J) ./ (2 .+ dt2_rho2)

    # Update current and electric field
    Jnpdemi = J .+ 0.5 * dt .* Enew .* rho
    Enp1 = E .- dt .* Jnpdemi

    # Semi-Lagrangian step
    for i in eachindex(Enew)
        ft = f[:, i]
        u1 = semilag_d2(ft, Enew[i], dt, dv)
        f[:, i] .= u1
    end

    return E .= Enp1
end
