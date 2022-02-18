function [D,Dt] = FD3D(k)


% Written by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016

%finite difference operators
%k is the order of the PA transform
D = @(U)ForwardD(U,k);
Dt = @(X,Y,Z)Dive(X,Y,Z,k);


%%%PA transform
function [X,Y,Z] = ForwardD(U,k)
[a,b,c] = size(U);
if k~=0
    if k<=size(U,2)
        X = diff(U,k,2);
        x = zeros(a,k,c);
        for i = 1:k
            x(:,i,:) = diff([U(:,end-k+i:end,:),zeros(a,i,c)],k,2);
        end
        X = [X,x];
    else
        X = zeros(a,b,c);
    end

    if k<=size(U,1)
        Y = diff(U,k,1);
        y = zeros(k,b,c);
        for i = 1:k
            y(i,:,:) = diff([U(end-k+i:end,:,:);zeros(i,b,c)],k,1);
        end
        Y = [Y;y];
    else
        Y = zeros(a,b,c);
    end


    if k<=size(U,3)
        Z = diff(U,k,3);
        z = zeros(a,b,k);
        for i = 1:k
            z(:,:,i) = diff([U(:,:,end-k+i:end);zeros(a,b,i)],k,1);
        end
        Z = cat(3,Z,z);
    else
        Z = zeros(a,b,c);
    end
else
    %standard l1 minimization of order is 0
    X=U;Y=U;Z=U;
end


%transpose PA transform
function dtxy = Dive(X,Y,Z,k)
[a,b,c] = size(X);

if k~=0
    if k <= b
        Dtx = (-1)^k*diff(X,k,2);
        dtx = zeros(a,k,c);       
        for i = 1:k
            dtx(:,i,:)= (-1)^k*diff([zeros(a,k-i+1,c),X(:,1:i,:)],k,2);
        end
        Dtx = [dtx,Dtx];
    else
        Dtx = 0;
    end



    if k <= a
        Dty = (-1)^k*diff(Y,k,1);
        dty = zeros(k,b,c);
        for i = 1:k
            dty(i,:,:) = (-1)^k*diff([zeros(k-i+1,b,c);Y(1:i,:,:)],k,1);
        end
        Dty = [dty;Dty];
    else
        Dty = 0;
    end


    if k<=c
        Dtz = (-1)^k*diff(Z,k,3);
        dtz = zeros(a,b,k);
        for i =1:k
            dtz(:,:,i) = (-1)^k*diff(cat(3,zeros(a,b,k-i+1),Z(:,:,1:i)),k,3);
        end

        Dtz = cat(3,dtz,Dtz);
    else
        Dtz = 0;
    end


    dtxy = Dty + Dtx + Dtz;
    dtxy = dtxy(:);
else
    %standard l1 minimization if order is 0
    dtxy = X(:)+Y(:)+Z(:);
end
    
    
    
    