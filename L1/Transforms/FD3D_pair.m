function [D,Dt] = FD3D_pair(ks,p,q,r,lambdas)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016

% finite difference operators for polynomial annihilation
% k is the order of the PA transform
D = @(U)ForwardD(U,ks,p,q,r,lambdas);
Dt = @(dU)Dive(dU,ks,p,q,r,lambdas);


% high order finite differences
function [dU] = ForwardD(U,ks,p,q,r,lambdas)
U = reshape(U,p,q,r);
dU = zeros(p*q*r,3,max(size(ks)));
c =0;


for k = ks
    if k~=0

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
    else
        %standard l1 minimization of order 0
        X=U;Y=U;Z=U;
    end
c = c+1;
dU(:,:,c) = [X(:),Y(:),Z(:)]*lambdas(c);

end


%transpose FD
function dtxy = Dive(dU,ks,p,q,r,lambdas)
dtxy = 0;
c = 0;
dU = reshape(dU,p*q*r,3,numel(ks));
for  k = ks
    c = c+1;
    X = reshape(dU(:,1,c),p,q,r);
    Y = reshape(dU(:,2,c),p,q,r);
    Z = reshape(dU(:,3,c),p,q,r);
    if k~=0
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


        dtxy = dtxy + (Dty(:) + Dtx(:) + Dtz(:))*lambdas(c);
        %dtxy = dtxy(:)*lambdas(c);
    else
        %standard l1 minimization if order 0
        dtxy = dtxy + (X(:)+Y(:)+Z(:))*lambdas(c);
    end

end