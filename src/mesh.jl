#=
function [ x,kk,kx,vxmin,vymin,vx,vy,Vx ] = mesh( kinit,Nx,Nvx,Nvy,vxmax,vymax)
%mesh in x,vx,vy

%mesh in x
xmin=0;
xmax=2*pi/kinit;
dx=(xmax-xmin)/Nx;
x=xmin:dx:xmax-dx;

%wavenumber in x
kk=2*pi/(xmax-xmin)*[0:Nx/2-1,Nx/2-Nx:Nx-1-Nx];
kx=(repmat(kk,Nvx,1))';

%mesh in v 
vxmin=-vxmax;vymin=-vymax;
dvx=2*vxmax/Nvx;
dvy=2*vymax/Nvy;
vx=(vxmin:dvx:vxmax-dvx)';
vy=(vymin:dvy:vymax-dvy)';

%dvx=(vxmax-vxmin)/(Nvx-1);
%dvy=(vymax-vymin)/(Nvy-1);
%vx=(vxmin:dvx:vxmax)';
%vy=(vymin:dvy:vymax)';
Vx=repmat(vx,1,Nx);

end

=#
