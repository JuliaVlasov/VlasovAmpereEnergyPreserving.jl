
#=
function diag = compute_diag(f,B,Ey,Ex,gy,vx,vy,ry,rx,Nx,Nvx,Nvy,dx,t)

    dvx=vx(2)-vx(1);dvy=vy(2)-vy(1);dxv=dx*dvx*dvy;
    kin=0;
    for k=1:Nx
        for i=1:Nvx-1
            for j=1:Nvy-1
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

function compute_diag(f, B, Ey, Ex, gy, vx, vy, ry, rx, Nx, Nvx, Nvy, dx, t)

    dvx=vx[2]-vx[1]
    dvy=vy[2]-vy[1]
    dxv=dx*dvx*dvy;
    kin=0;
    for k = 1:Nx
        for i = 1:(Nvx-1)
            for j = 1:(Nvy-1)
                normv=vx[i]*vx[i]+vy[j]*vy[j];
                kin=kin+normv*f[i, j, k];
            end
        end
    end
    kin=0.5*kin*dxv;

    diag[1]=t;
    diag[2]=dx*sum(B .* B);
    diag[3]=dx*sum(Ex .* Ex);
    diag[4]=dx*sum(Ey .* Ey);
    diag[5]=kin;
    diag[6]=sum(sum(sum(f)))*dxv;
    diag[7]=kin+rx^2+ry^2+0.5*dx*sum(B .* B);
    diag[8]=rx^2;
    diag[9]=ry^2;
    diag[10]=kin+0.5*dx*sum(Ex .* Ex)+0.5*dx*sum(Ey .* Ey)+0.5*dx*sum(B .* B);
end
