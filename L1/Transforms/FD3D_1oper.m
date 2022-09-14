function dU = FD3D(U,mode,k,p,q,r)


% finite difference operators for higher order TV
% k is the order of the transform
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016

switch mode
    case 1
        if k~=0
            U = reshape(U,p,q,r);
            dU = zeros(p,q,r,3);
            if k<=q
                dU(:,:,:,1) = diff([U,U(:,1:k,:)],k,2);
            end
            if k<=p
                dU(:,:,:,2) = diff([U;U(1:k,:,:)],k,1);
            end
            if k<=r
                dU(:,:,:,3) = diff(cat(3,U,U(:,:,1:k)),k,3);
            end
            dU = reshape(dU,p*q*r,3);
        else
            %standard l1 minimization of order 0
            dU = U(:);
        end
        
        dU = dU(:)*2^(1-k);  % normalization



%transpose FD
    case 2

        if k~=0
            dU = zeros(p,q,r);
            U = reshape(U,p,q,r,3);
            if k<=q
                dU = dU + (-1)^k*diff([U(:,end-k+1:end,:,1),U(:,:,:,1)],k,2);
            end    
            if k<=p
               dU = dU + (-1)^k*diff([U(end-k+1:end,:,:,2);U(:,:,:,2)],k,1);
            end
            if k<=r
                dU = dU + (-1)^k*diff(cat(3,U(:,:,end-k+1:end,3),U(:,:,:,3)),k,3);
            end
            dU = dU(:);
        else
            %standard l1 minimization
            dU = U(:);
        end
            
        dU = dU*2^(1-k);  % normalization
end
    
    
    