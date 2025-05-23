function init_f(Nx, Nvx, Nvy, x, vx, vy, kx, kk, testcase)
    dvx = vx[2] - vx[1]
    dvy = vy[2] - vy[1]

    ftab = zeros(Nvx, Nvy, Nx)

    if testcase == 1
        for i in 1:Nvx
            for j in 1:Nvy
                for k in 1:Nx
                    ftab[i,j,k] = (1/(2π)) * exp(-0.5*(vx[i]^2 + vy[j]^2)) * (1 + 0.001*cos(kx*x[k]))
                end
            end
        end
        Ex = zeros(1, Nx)
        Ey = zeros(1, Nx)
        B = zeros(1, Nx)
        rho = sum(ftab, dims=(1,2)) * dvx * dvy
        Ex = poisson(dropdims(rho, dims=(1,2)), x, kk)

    elseif testcase == 11
        for i in 1:Nvx
            for j in 1:Nvy
                for k in 1:Nx
                    ftab[i,j,k] = 1/(2π) * vx[i]^2 * exp(-0.5*(vx[i]^2 + vy[j]^2)) * (1 + 0.1*cos(kx*x[k]))
                end
            end
        end
        Ex = zeros(1, Nx)
        Ey = zeros(1, Nx)
        B = zeros(1, Nx)
        rho = sum(ftab, dims=(1,2)) * dvx * dvy
        Ex = poisson(dropdims(rho, dims=(1,2)), x, kk)
        B = 0.1*cos.(kx*x)

    elseif testcase == 2
        sigma1 = 0.02/sqrt(2)
        sigma2 = sqrt(12)*sigma1
        for i in 1:Nvx
            for j in 1:Nvy
                for k in 1:Nx
                    ftab[i,j,k] = (1/(2π*sigma1*sigma2)) *
                                 exp(-0.5*vx[i]^2/sigma1^2) *
                                 exp(-0.5*vy[j]^2/sigma2^2) * (1 + 0.0001*cos(kx*x[k]))
                end
            end
        end
        Ex = zeros(1, Nx)
        Ey = zeros(1, Nx)
        B = zeros(1, Nx)
        rho = sum(ftab, dims=(1,2)) * dvx * dvy
        Ex = poisson(dropdims(rho, dims=(1,2)), x, kk)
        B = 0.0001*cos.(kx*x)

    elseif testcase == 3
        sigma = 0.1/sqrt(2)
        v01 = 0.3
        v02 = -0.3
        delta = 1/2
        for i in 1:Nvx
            for j in 1:Nvy
                for k in 1:Nx
                    ftab[i,j,k] = (1/(2π*sigma^2)) * exp(-vx[i]^2/(2*sigma^2)) *
                                 (delta*exp(-(vy[j]-v01)^2/(2*sigma^2)) +
                                 (1-delta)*exp(-(vy[j]-v02)^2/(2*sigma^2)))
                end
            end
        end
        Ex = zeros(1, Nx)
        Ey = zeros(1, Nx)
        B = zeros(1, Nx)
        rho = sum(ftab, dims=(1,2)) * dvx * dvy
        Ex = poisson(dropdims(rho, dims=(1,2)), x, kk)
        B = 0.001*sin.(kx*x)

    elseif testcase == 4
        sigma1 = 1
        sigma2 = sqrt(6)*sigma1
        for i in 1:Nvx
            for j in 1:Nvy
                for k in 1:Nx
                    ftab[i,j,k] = (1/(2π*sigma2)) *
                                 exp(-0.5*vx[i]^2) *
                                 exp(-0.5*vy[j]^2/sigma2^2) * (1 + 1e-9*cos(kx*x[k]))
                end
            end
        end
        Ex = zeros(1, Nx)
        Ey = zeros(1, Nx)
        B = zeros(1, Nx)
        rho = sum(ftab, dims=(1,2)) * dvx * dvy
        Ex = poisson(dropdims(rho, dims=(1,2)), x, kk)
        B = 0.0001*cos.(kx*x)

    elseif testcase == 5
        sigma1 = 1
        sigma2 = sqrt(2)*sigma1
        for i in 1:Nvx
            for j in 1:Nvy
                for k in 1:Nx
                    ftab[i,j,k] = (1/(2π*sigma1*sigma2)) *
                                 exp(-0.5*vx[i]^2/sigma1^2) *
                                 exp(-0.5*vy[j]^2/sigma2^2) * (1 + 0.001*cos(kx*x[k]))
                end
            end
        end
        Ex = zeros(1, Nx)
        Ey = zeros(1, Nx)
        B = zeros(1, Nx)
        rho = sum(ftab, dims=(1,2)) * dvx * dvy
        Ex = poisson(dropdims(rho, dims=(1,2)), x, kk)
        B = 1e-6*cos.(kx*x)
    end

    return ftab, B, Ey, Ex
end
