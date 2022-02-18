function [T,Tt] = FDDD_2D(k,p,q,theta,w)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 09/10/2018

[D1,~] = FD3D(k,p,q,1);
if w>1 || w<0, error('w should be in [0,1]'); end
v = 1-w;
T = @(U)DD_Forward(D1,U,theta,w,v);
Tt = @(dU)DD_Adjoint(dU,theta,k,p,q,w,v);


function [dU] = DD_Forward(D1,U,theta,w,v)

dU = D1(U);
dU = [dU(:,1)*w*cosd(theta)+dU(:,2)*w*sind(theta),...
    dU(:,1)*v*sind(-theta)+dU(:,2)*v*cosd(theta)];

function U = DD_Adjoint(dU,theta,k,p,q,w,v)

dU = reshape(dU,p,q,2);
U = zeros(p,q);
Dxt = (-1)^k*diff([dU(:,end-k+1:end,:),dU],k,2);
Dyt = (-1)^k*diff([dU(end-k+1:end,:,:);dU],k,1);
U = cosd(theta)*(w*Dxt(:,:,1)+v*Dyt(:,:,2))+...
    sind(theta)*(w*Dyt(:,:,1)-v*Dxt(:,:,2));
U = U(:)*2^(1-k);




