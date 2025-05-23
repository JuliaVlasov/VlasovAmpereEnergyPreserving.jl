function [ fn,Ey ] = HB(fn,Ey,B,vx,vy,kk,kkvx,kkvy,Nvx,Nvy,dvx,dvy,Nx,dt,d,meth)
  
%solve the subsystem 
%df/dt+B Jv.df/dv=0 (with semi-Lagrangian + exact splitting) 
%dEy/dt=-dB/dx
%dB/dt=drx/dt=dry/dt=0

if (meth==0)
    fn=fft(fn,[],1);
    for l=1:Nx
      for j=1:Nvy
        for i=1:Nvx
            fn(i,j,l)=exp(-1i*kkvx(i)*vy(j)*tan(B(l)*dt/2)).*fn(i,j,l);
        end
      end
    end
    fn=ifft(fn,[],1);    
    
    fn=fft(fn,[],2);
    for l=1:Nx
     for i=1:Nvx
         for j=1:Nvy
             fn(i,j,l)=exp(1i*kkvy(j)*vx(i)*sin(B(l)*dt)).*fn(i,j,l);
         end
      end
    end
    fn=ifft(fn,[],2); 
    
    
    fn=fft(fn,[],1);
    for l=1:Nx
      for j=1:Nvy
        for i=1:Nvx
            fn(i,j,l)=exp(-1i*kkvx(i)*vy(j)*tan(B(l)*dt/2)).*fn(i,j,l);
        end
      end
    end
    fn=ifft(fn,[],1);  
    
end

if (meth==1)
    
    for j=1:Nvy
        for i=1:Nx
            ft=fn(:,j,i)';
            dep=vy(j)*(tan(B(i)*dt/2))/(dt/2);
            u1=semilag(ft,Nvx,dep,dt/2,dvx,d);
            fn(:,j,i)=u1;
        end
    end
    
    for j=1:Nvx
        for i=1:Nx
            ft=fn(j,:,i);
            dep=-vx(j)*(sin(B(i)*dt))/dt;
            u1=semilag(ft,Nvy,dep,dt,dvy,d);
            fn(j,:,i)=u1;
        end
    end

    for j=1:Nvy
        for i=1:Nx
            ft=fn(:,j,i)';
            dep=vy(j)*(tan(B(i)*dt/2))/(dt/2);
            u1=semilag(ft,Nvx,dep,dt/2,dvx,d);
            fn(:,j,i)=u1;
        end
    end
    
end


   amperey=real(ifft(sqrt(-1)*kk.*fft(B)));
   Ey=Ey-dt*amperey;
end







%     kkvx=2*pi/(2*vxmax)*[0:(Nvx/2),-(Nvx/2)+1:-1];
%     kkvy=2*pi/(2*vymax)*[0:(Nvy/2),-(Nvy/2)+1:-1];
% 
%     f2=fft(f2,[],1);
%     for j=1:Nvy
%         for i=1:Nvx
%        %fs=real(ifft(exp(-sqrt(-1)*vy.*kkvx*tan(dt/2)).*fft(f2(:,j))));
%             f2(i,j)=exp(-1i*kkvx(i)*vy(j)*tan(dt/2)).*f2(i,j);
%         end
%     end
%     f2=ifft(f2,[],1);
%     
%     f2=fft(f2,[],2);
%     for i=1:Nvx
%         for j=1:Nvy
%             f2(i,j)=exp(1i*kkvy(j)*vx(i)*sin(dt)).*f2(i,j);
%         end
%     end
%     f2=ifft(f2,[],2); 
%     
%     f2=fft(f2,[],1);
%     for j=1:Nvy
%         for i=1:Nvx
%        %fs=real(ifft(exp(-sqrt(-1)*vy.*kkvx*tan(dt/2)).*fft(f2(:,j))));
%             f2(i,j)=exp(-1i*kkvx(i)*vy(j)*tan(dt/2)).*f2(i,j);
%         end
%     end
%     f2=real(ifft(f2,[],1));
%     

