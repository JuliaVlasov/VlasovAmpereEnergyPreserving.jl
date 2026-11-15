using FFTW
using KernelAbstractions

@kernel function transport_velocity!(out, f, shifts, axis, spacing)
    i, j, l = @index(Global, NTuple)
    n = axis == 1 ? size(f, 1) : size(f, 2)
    displacement = -shifts[size(shifts, 1) == 1 ? 1 : i, size(shifts, 2) == 1 ? 1 : j, l] / spacing
    offset = floor(Int, displacement)
    α = displacement - offset
    index = axis == 1 ? i : j
    if iszero(α) || α == one(α)
        p = mod1(index + offset + (α == one(α) ? 1 : 0), n)
        out[i, j, l] = axis == 1 ? f[p, j, l] : f[i, p, l]
    else
        product = one(α)
        for m in -9:10
            product *= α - m
        end
        value = zero(eltype(f))
        for m in -9:10
            denom = one(α)
            for q in -9:10
                if m != q
                    denom *= m - q
                end
            end
            p = mod1(index + offset + m, n)
            sample = axis == 1 ? f[p, j, l] : f[i, p, l]
            value += (product / (α - m) / denom) * sample
        end
        out[i, j, l] = value
    end
end

function advect_velocity(f, shift, axis, dv)
    out = similar(f)
    backend = get_backend(f)
    transport_velocity!(backend)(out, f, shift, axis, dv; ndrange = size(f))
    return out
end

@kernel function linear_velocity!(out, f, shifts, axis, spacing)
    i, j, l = @index(Global, NTuple)
    n = axis == 1 ? size(f, 1) : size(f, 2)
    index = axis == 1 ? i : j
    coordinate = clamp(
        index - shifts[1, 1, l] / spacing, one(spacing), oftype(spacing, n),
    )
    lower = floor(Int, coordinate)
    upper = min(lower + 1, n)
    α = coordinate - lower
    left = axis == 1 ? f[lower, j, l] : f[i, lower, l]
    right = axis == 1 ? f[upper, j, l] : f[i, upper, l]
    out[i, j, l] = left + α * (right - left)
end

function advect_linear(f, shift, axis, dv)
    out = similar(f)
    linear_velocity!(get_backend(f))(out, f, shift, axis, dv; ndrange = size(f))
    return out
end

@inline function cubic_weight(t, n)
    if n == 0
        return (one(t) - t)^3 / 6
    elseif n == 1
        return (3t^3 - 6t^2 + 4) / 6
    elseif n == 2
        return (-3t^3 + 3t^2 + 3t + 1) / 6
    else
        return t^3 / 6
    end
end

@kernel function rotate_spline!(out, coefficients, B, vx, vy, vxmin, vymin, dvx, dvy, dt)
    i, j, l = @index(Global, NTuple)
    s, c = sincos(B[l] * dt)
    x = (c * vx[i] - s * vy[j] - vxmin) / dvx
    y = (s * vx[i] + c * vy[j] - vymin) / dvy
    ix, iy = floor(Int, x), floor(Int, y)
    tx, ty = x - ix, y - iy
    value = zero(eltype(coefficients))
    for a in 0:3, b in 0:3
        px = mod(ix + a - 1, size(coefficients, 1)) + 1
        py = mod(iy + b - 1, size(coefficients, 2)) + 1
        value += cubic_weight(tx, a) * cubic_weight(ty, b) * coefficients[px, py, l]
    end
    out[i, j, l] = value
end

function rotate_cubic(f, B, vx, vy, vxmin, vymin, dvx, dvy, dt, filterx, filtery)
    # Invert the periodic cubic B-spline interpolation operator in each velocity dimension.
    coefficients = real.(ifft(fft(f, (1, 2)) .* filterx .* filtery, (1, 2)))
    out = similar(f)
    rotate_spline!(get_backend(f))(
        out, coefficients, B, vx, vy, vxmin, vymin, dvx, dvy, dt; ndrange = size(f),
    )
    return out
end

function benchmark_parameters(case)
    if case == "weibel"
        return (; k = 1.25, α = 0.0, β = -1.0e-4, σ1 = 0.02 / sqrt(2),
                σ2 = sqrt(12) * 0.02 / sqrt(2), vxmax = 0.2, vymax = 0.5, tmax = 300.0)
    elseif case == "landau-weibel"
        return (; k = 0.5, α = 1.0e-3, β = -1.0e-6, σ1 = 1.0, σ2 = sqrt(2.0),
                vxmax = 6.0, vymax = 8.0, tmax = 50.0)
    end
    throw(ArgumentError("case must be weibel or landau-weibel"))
end

"""
Evolve the Section 6.2 or 6.3 initial condition on an array backend.

`array_type` is `Array` for CPU verification or `CuArray` for the CUDA run.
The default `:paper` mode uses linear electric transport and periodic
two-dimensional cubic B-spline magnetic transport. The paper does not specify
its B-spline boundary treatment. `:high_order` reproduces the interpolation
used by the existing Julia CPU example instead.
"""
function simulate_vm(case; nx, nv, dt, tmax = benchmark_parameters(case).tmax,
        array_type = Array, T = Float32, diagnostics = false, method = :paper)
    p = benchmark_parameters(case)
    nsteps = round(Int, tmax / dt)
    isapprox(nsteps * dt, tmax; atol = 100eps(Float64) * max(1, tmax)) ||
        throw(ArgumentError("tmax must be an integer multiple of dt"))
    isodd(nv) || throw(ArgumentError("nv must be odd: the Fourier current update divides by vx"))
    method in (:paper, :high_order) || throw(ArgumentError("method must be :paper or :high_order"))

    dt = T(dt)
    k = T(p.k)
    L = T(2π / p.k)
    dx = L / nx
    dvx = T(2p.vxmax / nv)
    dvy = T(2p.vymax / nv)
    xh = T.(0:(nx - 1)) .* dx
    vxh = T(-p.vxmax) .+ T.(0:(nv - 1)) .* dvx
    vyh = T(-p.vymax) .+ T.(0:(nv - 1)) .* dvy
    kxh = T.(2π / Float64(L) .* fftfreq(nx, nx))
    ikinv = Complex{T}[iszero(q) ? zero(Complex{T}) : inv(im * q) for q in kxh]
    vx = array_type(vxh)
    vy = array_type(vyh)
    kx = array_type(kxh)
    invik = array_type(ikinv)
    xv = reshape(array_type(xh), 1, 1, nx)
    vxv = reshape(vx, nv, 1, 1)
    vyv = reshape(vy, 1, nv, 1)
    filter = T.(3 ./ (2 .+ cos.(2π .* fftfreq(nv, nv) ./ nv)))
    filterx = reshape(array_type(filter), nv, 1, 1)
    filtery = reshape(array_type(filter), 1, nv, 1)
    f = @. exp(-T(0.5) * (vxv^2 / T(p.σ1)^2 + vyv^2 / T(p.σ2)^2)) *
        (1 + T(p.α) * cos(k * xv)) / T(2π * p.σ1 * p.σ2)
    rho = vec(sum(f; dims = (1, 2))) .* (dvx * dvy) .- one(T)
    Ex = real.(ifft(fft(rho) .* invik))
    Ey = array_type(zeros(T, nx))
    B = T(p.β) .* cos.(k .* array_type(xh))
    rx = sqrt(one(T) + T(0.5) * sum(abs2, Ex) * dx)
    ry = sqrt(one(T) + T(0.5) * sum(abs2, Ey) * dx)
    history = NamedTuple[]

    function record!(step)
        rho = vec(sum(f; dims = (1, 2))) .* (dvx * dvy) .- one(T)
        kinetic = T(0.5) * sum((vxv .^ 2 .+ vyv .^ 2) .* f) * dx * dvx * dvy
        magnetic = sum(abs2, B) * dx
        push!(history, (;
            time = Float64(step * dt),
            magnetic_norm = Float64(magnetic),
            ex_norm = Float64(sum(abs2, Ex) * dx),
            ey_norm = Float64(sum(abs2, Ey) * dx),
            total_energy = Float64(kinetic + rx^2 + ry^2 + T(0.5) * magnetic - T(2)),
            gauss_error = Float64(sqrt(sum(abs2, real.(ifft(im .* kx .* fft(Ex))) .- rho) * dx)),
        ))
    end
    diagnostics && record!(0)

    for step in 1:nsteps
        rho0 = vec(sum(f; dims = (1, 2))) .* (dvx * dvy)
        rhoy0 = vec(sum((vyv ./ vxv) .* f; dims = (1, 2))) .* (dvx * dvy)
        f = real.(ifft(fft(f, 3) .* exp.(-im .* vxv .* reshape(kx, 1, 1, nx) .* dt), 3))
        rho1 = vec(sum(f; dims = (1, 2))) .* (dvx * dvy)
        rhoy1 = vec(sum((vyv ./ vxv) .* f; dims = (1, 2))) .* (dvx * dvy)
        Ex = real.(ifft((fft(Ex) .+ fft(rho1 .- rho0) .* invik) .* (kx .!= 0)))
        Ey = real.(ifft((fft(Ey) .+ fft(rhoy1 .- rhoy0) .* invik) .* (kx .!= 0)))

        gx = Ex ./ rx
        gy = Ey ./ ry
        Jx = vec(sum(vxv .* f; dims = (1, 2))) .* (dvx * dvy)
        Jy = vec(sum(vyv .* f; dims = (1, 2))) .* (dvx * dvy)
        dxgy = real.(ifft(im .* kx .* fft(gy)))
        denomx = one(T) + dt^2 / T(8) * sum(rho1 .* gx .^ 2) * dx
        rx1 = (rx - dt / T(2) * sum(gx .* Jx) * dx -
               dt^2 / T(8) * rx * sum(rho1 .* gx .^ 2) * dx) / denomx
        denomy = one(T) + dt^2 / T(8) * (sum(rho1 .* gy .^ 2) + sum(dxgy .^ 2)) * dx
        ry1 = (ry - dt / T(2) * sum(gy .* Jy) * dx + dt / T(2) * sum(B .* dxgy) * dx -
               dt^2 / T(8) * ry * (sum(rho1 .* gy .^ 2) + sum(dxgy .^ 2)) * dx) / denomy
        Exmid = (rx1 + rx) / T(2) .* gx
        Eymid = (ry1 + ry) / T(2) .* gy
        B = B .- dt .* real.(ifft(im .* kx .* fft(Eymid)))
        if method == :paper
            f = advect_linear(f, reshape(dt .* Exmid, 1, 1, nx), 1, dvx)
            f = advect_linear(f, reshape(dt .* Eymid, 1, 1, nx), 2, dvy)
        else
            f = advect_velocity(f, reshape(dt .* Exmid, 1, 1, nx), 1, dvx)
            f = advect_velocity(f, reshape(dt .* Eymid, 1, 1, nx), 2, dvy)
        end
        rx, ry = rx1, ry1

        if method == :paper
            f = rotate_cubic(f, B, vx, vy, T(-p.vxmax), T(-p.vymax),
                             dvx, dvy, dt, filterx, filtery)
        else
            shearx = vyv .* reshape(tan.(B .* (dt / T(2))), 1, 1, nx)
            sheary = -vxv .* reshape(sin.(B .* dt), 1, 1, nx)
            f = advect_velocity(f, shearx, 1, dvx)
            f = advect_velocity(f, sheary, 2, dvy)
            f = advect_velocity(f, shearx, 1, dvx)
        end
        Ey = Ey .- dt .* real.(ifft(im .* kx .* fft(B)))
        diagnostics && record!(step)
    end

    KernelAbstractions.synchronize(get_backend(f))
    return (; f, Ex, Ey, B, rx, ry, history)
end
