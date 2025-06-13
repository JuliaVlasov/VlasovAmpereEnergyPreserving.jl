
#=
function diag = compute_diag(f,B,Ey,Ex,gy,vx,vy,ry,rx,nx,nvx,nvy,dx,t)

    dvx=vx(2)-vx(1);dvy=vy(2)-vy(1);dxv=dx*dvx*dvy;
    kin=0;
    for k=1:nx
        for i=1:nvx-1
            for j=1:nvy-1
                normv=vx(i)*vx(i)+vy(j)*vy(j);
                kin=kin+normv*f(i,j,k);
            end
        end
    end
    kin=0.5*kin*dxv;

    diag(1)=t;
    diag(2)=dx*sum(B.*B);
    diag(3)=dx*sum(Ex.*Ex);
    diag(4)=dx*sum(Ey.*Ey);
    diag(5)=kin;
    diag(6)=sum(sum(sum(f)))*dxv;
    diag(7)=kin+rx^2+ry^2+0.5*dx*sum(B.*B);
    diag(8)=rx^2;
    diag(9)=ry^2;
    diag(10)=kin+0.5*dx*sum(Ex.*Ex)+0.5*dx*sum(Ey.*Ey)+0.5*dx*sum(B.*B);
end
=#

export compute_diag

function compute_diag(f, Ex, Ey, B, vx, vy, rx, ry, dx, t)
    dvx=vx[2]-vx[1]
    dvy=vy[2]-vy[1]
    dxv=dx*dvx*dvy
    kin=0

    for k in axes(f, 3), i in eachindex(vx), j in eachindex(vy)
        kin += (vx[i]*vx[i]+vy[j]*vy[j]) * f[i, j, k]
    end

    kin=0.5*kin*dxv;

    diag = Float64[]
    push!(diag, t)
    push!(diag, dx*sum(B .* B))
    push!(diag, dx*sum(Ex .* Ex))
    push!(diag, dx*sum(Ey .* Ey))
    push!(diag, kin)
    push!(diag, sum(f)*dxv)
    push!(diag, kin+rx^2+ry^2+0.5*dx*sum(B .* B))
    push!(diag, rx^2)
    push!(diag, ry^2)
    push!(diag, kin+0.5dx*sum(Ex .* Ex)+0.5dx*sum(Ey .* Ey)+0.5dx*sum(B .* B))

    names = (
        "time",
        "BNorm",
        "ExNorm",
        "EyNorm",
        "KineticEnergy",
        "IntegralF",
        "TotalEnergy1",
        "RxSquare",
        "RySquare",
        "TotalEnergy2",
    )
    Dict(zip(names, diag))
end
