function [D,Dt] = FD2D(k,p,q,r)

% finite difference operators for higher order TV
% k is the order of the transform
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016



D = @(U)D_Forward(U,k,p,q,r);
Dt = @(Uc)D_Adjoint(Uc,k,p,q,r);


% high order finite differences
function [dU] = D_Forward(U,k,p,q,r)


if k~=0
    U = reshape(U,p,q,r);
    dU = zeros(p,q,r,2);
    if k<=q
        dU(:,1:end-k,:,1) = diff(U,k,2);
    end
    if k<=p
        dU(1:end-k,:,:,2) = diff(U,k,1);
    end
    dU = reshape(dU,(p)*(q)*r,2);
else
    %standard l1 minimization of order 0
    dU = U(:);
end

dU = dU*2^(1-k);  % normalization


%transpose FD
function U = D_Adjoint(dU,k,p,q,r)

if k~=0
    U = zeros(p,q,r);
    dU = reshape(dU,p,q,r,2);
    dU(:,end-k+1:end,:,1) = 0;
    dU(end-k+1:end,:,:,2) = 0;
    if k<=q
        U = U + (-1)^k*diff(cat(2,zeros(p,k,r),dU(:,:,:,1)),k,2);
     %   U = U + (-1)^k*cat(2,dU(:,1,:,1),diff(dU(:,1:end-k,:,1),k,2),-dU(:,end-1,:,1));
    end    
    if k<=p
        U = U + (-1)^k*diff(cat(1,zeros(k,q,r),dU(:,:,:,2)),k,1);
   %      U = U + (-1)^k*cat(1,dU(1,:,:,2),diff(dU(1:end-k,:,:,2),k,1),-dU(end-1,:,:,2));
    end
    U = U(:);
else
    %standard l1 minimization
    U = dU(:);
end
    
U = U*2^(1-k);  % normalization
    
    
    