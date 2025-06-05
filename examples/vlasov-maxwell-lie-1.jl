using FFTW
using LinearAlgebra
using Printf
using BenchmarkTools

function main()
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
    rx = sqrt(0.5 * sum(Ex .^ 2) * dx + 1.0)
    ry = sqrt(0.5 * sum(Ey .^ 2) * dx + 1.0)
    gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)

    # Initial diagnostics
    diag_tmp = compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, Nx, Nvx, Nvy, dx, 0.0)
    en_init = diag_tmp[7]
    diag[:, 1] = diag_tmp
    verif_poisson = [0.0]

    # Useful arrays
    kkvx = 2π / (2 * vxmax) * vcat(0:((Nvx-1)/2), (-(Nvx+1)/2+1):-1)
    kkvy = 2π / (2 * vymax) * vcat(0:((Nvy-1)/2), (-(Nvy+1)/2+1):-1)
    X = repeat(x', Nvx, 1)
    kkv = 2π / (vxmax - vxmin) * vcat(0:floor(Int, Nvx/2), (-floor(Int, Nvx/2)):-1)
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

        hf!(f, Ex, Ey, dt, dvx, dvy, Nvx, Nvy, Nx, Vx, vy, kx, kk)
        gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)

        HE(f, rx, ry, B, vx, vy, gx, gy, Nvx, Nvy, kk, kv, dvx, dvy, Nx, dx, dt)
        gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)

        HB!(f, Ey, B, vx, vy, kk, kkvx, kkvy, Nvx, Nvy, dvx, dvy, Nx, dt)
        gx, gy = update_g(rx, Ex, ry, Ey, x, Nx)

    end

end



# Run the simulation
@time results = main()
