function [D,Dt] = FD2D_pair(ks,p,q,lambdas)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016

% finite difference operators for polynomial annihilation
% k is the order of the PA transform
D = @(U)ForwardD(U,ks,p,q,lambdas);
Dt = @(dU)Dive(dU,ks,p,q,lambdas);


% high order finite differences
function [dU] = ForwardD(U,ks,p,q,lambdas)
U = reshape(U,p,q);
nzeroK = sum(ks>0);
zeroK = sum(ks==0);
dU = zeros(p*q,2*nzeroK+zeroK);

cnt = 0;
for i = 1:numel(ks)
    k = ks(i);
    if k~=0

        if k<=size(U,2)
            X = diff([U,U(:,1:k)],k,2);
        else
            X = zeros(size(U));
        end

        if k<=size(U,1)
            Y = diff([U;U(1:k,:)],k,1);
        else
            Y = zeros(size(U));
        end
        dU(:,cnt+1:cnt+2) = [X(:),Y(:)]*lambdas(i);
        cnt = cnt+2;
    else
        %standard l1 minimization of order 0
        % X=U;Y=U;
        dU(:,cnt+1) = U(:)*lambdas(i);
        cnt = cnt+1;
    end
end
dU = dU(:);

%transpose FD
function dtxy = Dive(dU,ks,p,q,lambdas)
dtxy = 0;
cnt = numel(dU)/p/q;
dU = reshape(dU,p*q,cnt);

cnt  = 0;
for  i = 1:numel(ks)
    k = ks(i);
    if k~=0
        X = reshape(dU(:,cnt+1),p,q);
        Y = reshape(dU(:,cnt+2),p,q);
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

        dtxy = dtxy + (Dty(:) + Dtx(:))*lambdas(i);
        cnt = cnt + 2;
    else
        %standard l1 minimization if order 0
        dtxy = dtxy + dU(:,cnt+1)*lambdas(i);
        cnt = cnt+1;

    end

end