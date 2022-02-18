function [D,Dt] = FDZ(k,p,q,r)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/3/2016


% finite difference operators for polynomial annihilation
% k is the order of the PA transform
D = @(U)D_Forward(U,k,p,q,r);
Dt = @(Uc)D_Adjoint(Uc,k,p,q,r);


function [dU] = D_Forward(U,k,p,q,r)
if k~=0
    U = reshape(U,p,q,r);
    dU = diff(U,k,3);
else
    dU = U(:);
end

dU = dU(:)*2^(1-k);


function U = D_Adjoint(dU,k,p,q,r)

if k~=0
    dU = reshape(dU,p,q,r-k);
    U = (-1)^k*diff(cat(3,zeros(p,q,k),dU,zeros(p,q,k)),k,3);
else
    U = dU(:);
end

U = U(:)*2^(1-k);