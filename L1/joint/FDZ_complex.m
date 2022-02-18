function [D,Dt] = FDZ_complex(k,p,q,r,theta)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016


% finite difference operators for HOTV
% k is the order of the HOTV transform
D = @(U)D_Forward(U,k,p,q,r,theta);
Dt = @(Uc)D_Adjoint(Uc,k,p,q,r,theta);


function [dU] = D_Forward(U,k,p,q,r,theta)
if k~=0
    U = reshape(U,p,q,r).*theta;
    
    dU = diff(U,k,3);
else
    dU = U(:);
end

dU = dU(:)*2^(1-k);


function U = D_Adjoint(dU,k,p,q,r,theta)

if k~=0
    dU = reshape(dU,p,q,r-k);
    U = (-1)^k*diff(cat(3,zeros(p,q,k),dU,zeros(p,q,k)),k,3);
else
    U = dU(:);
end

U = U(:)*2^(1-k).*conj(theta(:));