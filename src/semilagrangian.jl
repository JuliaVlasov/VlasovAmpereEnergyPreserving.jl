function semilag(u, N, a, dt, dx, d)
    # semi-Lagrangian method for u_t + a u_x = 0 + periodic boundary conditions
    # u1(x) = u(x-a*dt) with Lagrange interpolation of degree 2d+1
    
    u1 = zeros(N)
    
    for i in 1:N
        j = floor(Int, i - a*dt/dx)
        α = (i - a*dt/dx) - j
        
        # Initialize arrays
        j_ind = zeros(Int, 2d+2)
        Lag = zeros(2d+2)
        
        # Calculate indices with periodic boundary conditions
        for ii in -d:(d+1)
            j_ind[ii+d+1] = periodize(j + ii, N+1)
        end
        
        # Initialize matrices
        P = ones(2d+2, 2d+2)
        C = ones(2d+2, 2d+2)
        
        # Fill P and C matrices
        for ii in -d:(d+1), jj in -d:(d+1)
            if ii ≠ jj
                P[jj+d+1, ii+d+1] = (α - ii)
            end
            C[jj+d+1, ii+d+1] = (jj - ii)
        end
        
        # Calculate Lagrange coefficients
        for ii in -d:(d+1)
            Lag[ii+d+1] = prod(P[ii+d+1, :]) / prod(filter(!iszero, C[ii+d+1, :]))
        end
        
        # Compute interpolated value
        u1[i] = sum(Lag .* u[j_ind])
    end
    
    return u1
end

# Assuming you have a periodize function defined in Julia
function periodize(index, N)
    # Implement your periodic boundary condition logic here
    # Example implementation:
    mod(index-1, N) + 1
end
