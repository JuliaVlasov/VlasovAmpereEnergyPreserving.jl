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

export HE_lapenta

function HE_lapenta(fn, V, E, Nv, kv, dv, Nx, dx, dt, d)
    # Calculate current and charge density
    J = vec(sum(V .* fn; dims=1) .* dv)
    rho = vec(sum(fn; dims=1) .* dv)

    # Update electric field
    Enew = (1 ./ (1 .+ (dt^2 .* rho ./ 4))) .* E .- (dt ./ (2 .+ (dt^2 .* rho ./ 2))) .* J

    # Update current
    Jnp1 = J .+ dt .* Enew .* rho
    Jnpdemi = J .+ 0.5 .* dt .* Enew .* rho
    Enp1 = E .- dt .* Jnpdemi

    # Initialize output array
    fn1 = similar(fn)

    # Apply semi-Lagrangian method to each spatial point
    for i in 1:Nx
        ft = fn[:, i]'  # Transpose to row vector
        u1 = semilag(ft, Nv, Enew[i], dt, dv, d)
        fn1[:, i] = u1
    end

    return fn1, Enp1
end

function HE_lapenta(
    fn::Matrix{T},
    V::Matrix{T},
    E::Vector{T},
    Nv::Int,
    kv::Vector{T},
    dv::T,
    Nx::Int,
    dx::T,
    dt::T,
    d::Int,
) where {T<:AbstractFloat}

    # Pre-allocate outputs
    fn1 = similar(fn)
    Enp1 = similar(E)

    # Calculate current and charge density (using views to avoid allocations)
    J = @views sum(V .* fn, dims=1) .* dv
    rho = @views sum(fn, dims=1) .* dv

    # Compute intermediate terms (using fused broadcasting)
    dt2_rho4 = (dt^2) .* rho ./ 4
    dt2_rho2 = dt2_rho4 .* 2  # Same as dt^2*rho/2

    # Update electric field (single fused operation)
    @. Enew = E / (1 + dt2_rho4) - (dt * J) / (2 + dt2_rho2)

    # Update current and electric field
    @. Jnpdemi = J + 0.5 * dt * Enew * rho
    @. Enp1 = E - dt * Jnpdemi

    # Semi-Lagrangian step (multi-threaded)
    @inbounds Threads.@threads for i in 1:Nx
        ft = @view fn[:, i]  # Column vector view
        u1 = semilag(ft, Nv, Enew[i], dt, dv, d)
        fn1[:, i] = u1
    end

    return fn1, Enp1
end
