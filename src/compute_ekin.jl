function kin = compute_ekin(f,vx,vy,Nx,Nvx,Nvy,dx)

    dvx=vx(2)-vx(1);dvy=vy(2)-vy(1);
    for k=1:Nx
        kin(k)=0;
        for i=1:Nvx-1
            for j=1:Nvy-1
                normv=0.5*(vx(i)*vx(i)+vy(j)*vy(j));
                kin(k)=kin(k)+normv*f(i,j,k)*dvx*dvy;
            end
        end
    end
    
end

