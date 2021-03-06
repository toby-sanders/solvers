function [D,Dt] = FD2D(k,p,q)

% finite difference operators for higher order TV
% k is the order of the transform
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016



D = @(U)D_Forward(U,k,p,q);
Dt = @(Uc)D_Adjoint(Uc,k,p,q);


% high order finite differences
function [dU] = D_Forward(U,k,p,q)


if k~=0
    % U = reshape(U,p,q);
    dU = zeros(p,q,2);
    if k<=q
        dU(:,:,1) = diff([U,U(:,1:k)],k,2);
    end
    if k<=p
        dU(:,:,2) = diff([U;U(1:k,:)],k,1);
    end
    % dU = reshape(dU,p*q,2);
else
    %standard l1 minimization of order 0
    dU = U;
    % dU = U(:);
end

dU = dU*2^(1-k);  % normalization



%transpose FD
function U = D_Adjoint(dU,k,p,q)

if k~=0
    U = zeros(p,q);
    % dU = reshape(dU,p,q,2);
    if k<=q
        U = U + (-1)^k*diff([dU(:,end-k+1:end,1),dU(:,:,1)],k,2);
    end    
    if k<=p
        U = U + (-1)^k*diff([dU(end-k+1:end,:,2);dU(:,:,2)],k,1);
    end
    % U = U(:);
else
    %standard l1 minimization
    U = dU;% reshape(dU,p,q);
end
    
U = U*2^(1-k);  % normalization
    
    
    