function make_semilag(d::Int)
    @inline function specialized_semilag(
            f::Vector{Float64}, E::Float64, dt::Float64, dv::Float64
        )::Vector{Float64}
        # Initialize output and working arrays
        f_new = similar(f)
        alpha = zeros(2d + 2)
        j_ind = Vector{Int}(undef, 2d + 2)

        # Precompute the shift amount
        shift = E * dt / dv
        Nv = length(f)

        # Main loop over velocity grid points
        @inbounds for i in eachindex(f_new)
            # Compute base index and fractional part
            j = floor(Int, i - shift)
            α = (i - shift) - j

            # Periodic boundary handling (optimized)
            j = mod1(j, Nv)

            # Determine interpolation stencil indices
            @simd for ii in (-d):(d + 1)
                j_ind[ii + d + 1] = mod1(j + ii, Nv)
            end

            # Compute Lagrange coefficients
            P = ones(2d + 2, 2d + 2)
            C = ones(2d + 2, 2d + 2)

            @fastmath for ii in (-d):(d + 1), jj in (-d):(d + 1)
                if ii ≠ jj
                    P[jj + d + 1, ii + d + 1] = (α - ii)
                end
                C[jj + d + 1, ii + d + 1] = (jj - ii)
            end

            # Compute Lagrange weights
            L = zeros(2d + 2)
            @fastmath for ii in (-d):(d + 1)
                prodP = 1.0
                prodC = 1.0
                for jj in (-d):(d + 1)
                    if jj ≠ ii
                        prodP *= P[ii + d + 1, jj + d + 1]
                        prodC *= C[ii + d + 1, jj + d + 1]
                    end
                end
                L[ii + d + 1] = prodP / prodC
            end

            # Apply interpolation
            f_new[i] = zero(Float64)
            @simd for ii in 1:(2d + 2)
                f_new[i] += L[ii] * f[j_ind[ii]]
            end
        end

        return f_new
    end

    return specialized_semilag
end

# Precompile for common degrees
const semilag_d1 = make_semilag(1)
const semilag_d2 = make_semilag(2)
const semilag_d3 = make_semilag(3)
const semilag_d9 = make_semilag(9)
