iusing FFTW
using LinearAlgebra
using Printf
using BenchmarkTools

function vm_1d2v_simulation()
    # Start timer
    start_time = time()

    # Computational domain parameters
    Nx = 16
    Nvx = 47
    Nvy = 47
    dt = 0.125 / 2
    tmax = 300.0
    Nt = floor(Int, tmax / dt)

    # Test case selection
    testcase = 2
    kinit = 1.25
    vxmax = 0.2
    vymax = 0.5

    # Initialize arrays
    finit = zeros(Nvx, Nvy, Nx)
    f = zeros(Nvx, Nvy, Nx)
    f1 = zeros(Nvx, Nvy, Nx)
    f2 = zeros(Nvx, Nvy, Nx)
    f3 = zeros(Nvx, Nvy, Nx)
    tr_mat = zeros(Nvx, Nvy, Nx)
    diag = zeros(10, Nt+1)
    diag_tmp = zeros(10)
    screen_var = zeros(3)

    # Create phase space meshes
    x, kk, kx, vxmin, vymin, vx, vy, Vx = create_mesh(kinit, Nx, Nvx, Nvy, vxmax, vymax)

    # Initial conditions
    finit, B, Ey, Ex = init_f(Nx, Nvx, Nvy, x, vx, vy, kinit, kk, testcase)
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]
    dx = x[2] - x[1]
    f = copy(finit)
    rhoinit = sum(f) * dvx * dvy
    Exinit = copy(Ex)
    Binit = copy(B)
    Eyinit = copy(Ey)
    rx = sqrt(0.5 * sum(Ex.^2) * dx + 1.0)
    ry = sqrt(0.5 * sum(Ey.^2) * dx + 1.0)
    gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)

    # Initial diagnostics
    diag_tmp = compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, Nx, Nvx, Nvy, dx, 0.0)
    en_init = diag_tmp[7]
    diag[:, 1] = diag_tmp
    verif_poisson = [0.0]

    # Useful arrays
    kkvx = 2π / (2 * vxmax) * vcat(0:(Nvx-1)/2, -(Nvx+1)/2+1:-1)
    kkvy = 2π / (2 * vymax) * vcat(0:(Nvy-1)/2, -(Nvy+1)/2+1:-1)
    X = repeat(x', Nvx, 1)
    kkv = 2π / (vxmax - vxmin) * vcat(0:floor(Int, Nvx/2), -floor(Int, Nvx/2):-1)
    kv = repeat(kkv, 1, Nx)
    Vy = repeat(vy, 1, Nx)

    # Interpolation parameters
    meth = 0
    d = 1
    split = 1  # Splitting method

    # Time stepping
    t = 0.0
    iter = 0

    while abs(t - tmax) > 1e-10
        iter += 1
        t += dt

        if mod(iter, 50) == 0
            screen_var[1] = t
            @printf("Time: %.2f, Max gy: %.4e\n", t, maximum(abs.(gy)))
        end

        if split == 0
            # Hongtao-Lapenta splitting
            f = hf_lapenta2d(dt, Nvy, Vx, kx, f)
            f, Ex, Ey = HE_lapenta2d(f, vx, vy, Ex, Ey, Nvx, Nvy, dvx, dvy, Nx, dx, dt, d)
            f, Ex, Ey, B = HB_lapenta2d(f, Ex, Ey, B, vx, vy, kk, kkvx, kkvy, Nvx, Nvy, dvx, dvy, Nx, dt, dx, d, meth)
        elseif split == 1
            # Lie splitting (SAV 1st splitting)
            f, Ex, Ey = hf(dt, dvx, dvy, Nvx, Nvy, Nx, Vx, vy, kx, kk, f, Ex, Ey)
            gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)
            
            f, rx, ry, B = HE(f, vx, vy, Vx, Vy, rx, gx, ry, gy, B, Nvx, Nvy, kk, kv, dvx, dvy, Nx, dx, dt, d)
            gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)
            
            f, Ey = HB(f, Ey, B, vx, vy, kk, kkvx, kkvy, Nvx, Nvy, dvx, dvy, Nx, dt, d, meth)
            gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)
        elseif split == 3
            # Lie splitting (SAV 2nd splitting)
            f, Ex, Ey = hf(dt, dvx, dvy, Nvx, Nvy, Nx, Vx, vy, kx, kk, f, Ex, Ey)
            gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)
            
            f, rx, ry, gx, gy = HE2(f, vx, vy, rx, gx, Ex, ry, gy, Ey, Nvx, Nvy, dvx, dvy, Nx, dx, x, dt, d)
            gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)
            
            f, Ey, ry, B = HB2(f, vx, vy, Ey, ry, gy, B, Nvx, Nvy, kk, kkvx, kkvy, dvx, dvy, Nx, dx, dt, d, meth)
            gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)
        end

        # Diagnostics
        diag_tmp = compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, Nx, Nvx, Nvy, dx, t)
        rho = sum(f) * dvx * dvy
        Ex_poisson = poisson(dropdims(sum(f, dims=(1,2)), x, kk)
        push!(verif_poisson, norm(Ex_poisson - Ex))
        diag[:, iter+1] = diag_tmp
    end

    # End timer and print results
    elapsed_time = time() - start_time
    @printf("\nSimulation completed in %.2f seconds\n", elapsed_time)
    
    return diag, verif_poisson
end

# Run the simulation
diag, verif_poisson = vm_1d2v_simulation()
