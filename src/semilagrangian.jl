#=
function [ u1 ] = semilag( u,N,a,dt,dx, d )
%semi-Lagrangian method for u_t + a u_x = 0 + periodic boundary conditions
% u1(x) = u(x-a*dt) with Lagrange interpolation of degree 2d+1


for i=1:N

        j=floor(i-a*dt/dx);
        alpha=(i-a*dt/dx)-j;  


        for ii=-d:(d+1)  
            j_ind(ii+d+1)=periodize(j+ii,N+1);
        end

        P=ones(2*d+2,2*d+2);C=ones(2*d+2,2*d+2);
        for ii=-d:(d+1)
            for jj=-d:(d+1)
                if (ii~=jj)
                    P(jj+d+1,ii+d+1)=(alpha-ii);
                end
                C(jj+d+1,ii+d+1)=(jj-ii);
            end
        end


        for ii=-d:(d+1)
            Lag(ii+d+1)=prod(P(ii+d+1,:))/(prod(nonzeros(C(ii+d+1,:))));            
        end

        u1(i)=sum(Lag.*u(j_ind));
end
=#

function semilag(u, N, a, dt, dx, d)
    # semi-Lagrangian method for u_t + a u_x = 0 + periodic boundary conditions
    # u1(x) = u(x-a*dt) with Lagrange interpolation of degree 2d+1

    u1 = zeros(N)

    for i = 1:N
        j = floor(Int, i - a*dt/dx)
        α = (i - a*dt/dx) - j

        # Initialize arrays
        j_ind = zeros(Int, 2d+2)
        Lag = zeros(2d+2)

        # Calculate indices with periodic boundary conditions
        for ii = (-d):(d+1)
            j_ind[ii+d+1] = periodize(j + ii, N+1)
        end

        # Initialize matrices
        P = ones(2d+2, 2d+2)
        C = ones(2d+2, 2d+2)

        # Fill P and C matrices
        for ii = (-d):(d+1), jj = (-d):(d+1)
            if ii ≠ jj
                P[jj+d+1, ii+d+1] = (α - ii)
            end
            C[jj+d+1, ii+d+1] = (jj - ii)
        end

        # Calculate Lagrange coefficients
        for ii = (-d):(d+1)
            Lag[ii+d+1] = prod(P[ii+d+1, :]) / prod(filter(!iszero, C[ii+d+1, :]))
        end

        # Compute interpolated value
        u1[i] = sum(Lag .* u[j_ind])
    end

    return u1
end

#=
function [ j ] = periodize( j,N )

% if (j==0)
%     j=N-1;
% end
%
% if (j==-1)
%     j=N-2;
% end
while (j<=0)
    j=j+N-1;
end

while (j>N)
    j=j-N+1;
end

if (j==N)
    j=1;
end

end
=#

function periodize(j::Int, N::Int)::Int
    # Fast periodic boundary condition for 1-based indexing
    j = mod1(j, N-1)  # Handles most cases in one operation

    # Special case when j == N (mod1 would return N, but we want 1)
    if j == N
        return 1
    end

    return j
end

# @inline function periodize(j::Int, N::Int)::Int
#     j = Base.unsafe_mod1(j, N-1)
#     ifelse(j == N, 1, j)
# end

@inline function semilag(f::Vector{Float64}, Nv::Int, E::Float64, 
                        dt::Float64, dv::Float64, d::Int)::Vector{Float64}
    # Initialize output and working arrays
    f_new = similar(f)
    alpha = zeros(2d+2)
    j_ind = Vector{Int}(undef, 2d+2)
    
    # Precompute the shift amount
    shift = E * dt / dv
    
    # Main loop over velocity grid points
    @inbounds for i in 1:Nv
        # Compute base index and fractional part
        j = floor(Int, i - shift)
        α = (i - shift) - j
        
        # Periodic boundary handling (optimized)
        j = mod1(j, Nv)
        
        # Determine interpolation stencil indices
        @simd for ii in -d:(d+1)
            j_ind[ii+d+1] = mod1(j + ii, Nv)
        end
        
        # Compute Lagrange coefficients
        P = ones(2d+2, 2d+2)
        C = ones(2d+2, 2d+2)
        
        @fastmath for ii in -d:(d+1), jj in -d:(d+1)
            if ii ≠ jj
                P[jj+d+1, ii+d+1] = (α - ii)
            end
            C[jj+d+1, ii+d+1] = (jj - ii)
        end
        
        # Compute Lagrange weights
        L = zeros(2d+2)
        @fastmath for ii in -d:(d+1)
            prodP = 1.0
            prodC = 1.0
            for jj in -d:(d+1)
                if jj ≠ ii
                    prodP *= P[ii+d+1, jj+d+1]
                    prodC *= C[ii+d+1, jj+d+1]
                end
            end
            L[ii+d+1] = prodP / prodC
        end
        
        # Apply interpolation
        f_new[i] = zero(Float64)
        @simd for ii in 1:(2d+2)
            f_new[i] += L[ii] * f[j_ind[ii]]
        end
    end
    
    return f_new
end

function make_semilag(d::Int)
    @inline function specialized_semilag(f::Vector{Float64}, Nv::Int, E::Float64,
                                       dt::Float64, dv::Float64)::Vector{Float64}
        # Implementation specialized for degree d
        # ... same structure as above but with d hardcoded ...
    end
    return specialized_semilag
end

# Precompile for common degrees
const semilag_d1 = make_semilag(1)
const semilag_d2 = make_semilag(2)
const semilag_d3 = make_semilag(3)
