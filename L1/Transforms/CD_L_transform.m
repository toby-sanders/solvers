function [D,Dt] = CD_L_transform(nF,d)

D = @(U)D_Forward(U,nF,d);
Dt = @(Uc)D_Adjoint(Uc,nF,d);

% remark: I have found analytically that D^T D can be implemented
% (possibly) much faster than D, then D^T.  In future one may implement a 
% DD^T operator to avoid expensive memory and computation of Du.
% also, can observe this through matrix operations of DD^T or just basic
% derivatives of the objective function in sum form...


% forward operator
function [dU] = D_Forward(U,nF,d)
U = reshape(U,d,nF);
dU = zeros(d,nF,nF);
for i = 1:nF
    for j = 1:nF
        dU(:,j,i) = U(:,i) - U(:,j);
    end
end
dU = dU(:);

% transpose
function U = D_Adjoint(dU,nF,d)
dU = reshape(dU,d,nF,nF);
U = zeros(d,nF);
for i = 1:nF
    for j = 1:nF
        if i==j
            U(:,j) = U(:,j) + sum(dU(:,:,i),2)-dU(:,i,i);
        else
            U(:,j) = U(:,j) - dU(:,j,i);
        end
    end
end

U = U(:);
    
    