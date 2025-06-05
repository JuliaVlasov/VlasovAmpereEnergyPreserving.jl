
#=
function [ gx,gy ] = updateg(rx,Ex,ry,Ey,x,Nx )

        xmax=x(2)+x(end);
        k=2*pi/xmax;

       if (abs(rx)>1.e-12)
           gx=Ex/rx;
       else
           gx=zeros(1,Nx);%cos(k*x);%
       end

       if (abs(ry)>1.e-12)
           gy=Ey/ry;
       else
           gy=zeros(1,Nx);%cos(k*x);%
       end 

end
=#

function updateg(rx, Ex, ry, Ey, x)

    nx = length(x)
    xmax=x[2]+x[end]
    k=2pi/xmax

    if abs(rx)>1.e-12
        gx=Ex/rx
    else
        gx=zeros(nx)
    end

    if abs(ry)>1.e-12
        gy=Ey/ry
    else
        gy=zeros(nx)
    end

    return gx, gy

end
