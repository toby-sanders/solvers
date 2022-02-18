function [D,Dt] = FD3D(k,p,q,r)


% Written by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016


% finite difference operators for polynomial annihilation
% k is the order of the PA transform
D = @(U)D_Forward(U,k,p,q,r);
Dt = @(Uc)D_Adjoint(Uc,k,p,q,r);


% high order finite differences
function [dU] = D_Forward(U,k,p,q,r)


if k~=0
    U = reshape(U,p,q,r);
    if k<=size(U,2)
        X = diff([U,U(:,1:k,:)],k,2);
    else
        X = zeros(size(U));
    end

    if k<=size(U,1)
        Y = diff([U;U(1:k,:,:)],k,1);
    else
        Y = zeros(size(U));
    end


    if k<=size(U,3)
        Z = diff(cat(3,U,U(:,:,1:k)),k,3);
    else
        Z = zeros(size(U));
    end
    dU = [X(:),Y(:),Z(:)];
else
    %standard l1 minimization of order 0
    dU = U(:);
end



%transpose FD
function dtxy = D_Adjoint(dU,k,p,q,r)

if k~=0
    X = reshape(dU(:,1),p,q,r);
    Y = reshape(dU(:,2),p,q,r);
    Z = reshape(dU(:,3),p,q,r);
    if k<=size(X,2)
        Dtx = (-1)^k*diff([X(:,end-k+1:end,:),X],k,2);
    else
        Dtx = 0;
    end
    
    if k<=size(Y,1)
        Dty = (-1)^k*diff([Y(end-k+1:end,:,:);Y],k,1);
    else
        Dty = 0;
    end


    if k<=size(Z,3)
        Dtz = (-1)^k*diff(cat(3,Z(:,:,end-k+1:end),Z),k,3);
    else
        Dtz = 0;
    end


    dtxy = Dty + Dtx + Dtz;
    dtxy = dtxy(:);%*2^(1-k);
else
    %standard l1 minimization
    dtxy = dU(:);
end
    
    
    
    