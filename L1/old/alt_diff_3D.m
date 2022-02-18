function [D,Dt] = alt_diff_3D(k)


%FD that takes U_{xy} (multidirectional derivative)

%Written by: Toby Sanders
%Comp. & Applied Math Dept., Univ. of South Carolina
%Dept. of Math and Stat Sciences, Arizona State University
%02/26/2016


%finite difference operators for polynomial annihilation
%k is the order of the PA transform
D = @(U)ForwardD(U,k);
Dt = @(X,Y,Z)Dive(X,Y,Z,k);


%%%PA transform
function [X,Y,Z] = ForwardD(U,k)
if k~=0
    if k<=size(U,2)
        X = diff(U,k,2);
        x = zeros(size(U,1),k,size(U,3));
        for i = 1:k
            x(:,i,:) = diff([U(:,end-k+i:end,:),U(:,1:i,:)],k,2);
        end
        X = [X,x];
    else
        X = zeros(size(U));
    end

    if k<=size(U,1)
        Y = diff(X,k,1);
        y = zeros(k,size(X,2),size(X,3));
        for i = 1:k
            y(i,:,:) = diff([X(end-k+i:end,:,:);X(1:i,:,:)],k,1);
        end
        Y = [Y;y];
    else
        Y = zeros(size(U));
    end
    X = Y;

    if k<=size(U,3)
        Z = diff(U,k,3);
        z = zeros(size(U,1),size(U,2),k);
        for i = 1:k
            temp = cat(3,U(:,:,end-k+i:end),U(:,:,1:i));
            z(:,:,i) = diff(temp,k,3);
        end
        Z = cat(3,Z,z);
    else
        Z = zeros(size(U));
    end
    
    %Xt = X;
    %[X,~,~] = ForwardD(Y,k);
    %[~,Y,~] = ForwardD(Xt,k);
    
else
    %standard l1 minimization of order is 0
    X=U;Y=U;Z=U;
end


%transpose PA transform
function dtxy = Dive(X,Y,Z,k)
if k~=0
    if k<=size(X,2)
        Dtx = (-1)^k*diff(X,k,2);
        dtx = zeros(size(X,1),k,size(X,3));
        for i = 1:k
            dtx(:,i,:)= (-1)^k*diff([X(:,end-k+i:end,:),X(:,1:i,:)],k,2);
        end
        Dtx = [dtx,Dtx];
    else
        Dtx = 0;
    end



    if k<=size(Y,1)
        Dty = (-1)^k*diff(Dtx,k,1);
        dty = zeros(k,size(Dtx,2),size(Dtx,3));
        for i = 1:k
            dty(i,:,:) = (-1)^k*diff([Dtx(end-k+i:end,:,:);Dtx(1:i,:,:)],k,1);
        end

        Dty = [dty;Dty];
    else
        Dty = 0;
    end


    if k<=size(Z,3)
        Dtz = (-1)^k*diff(Z,k,3);
        dtz = zeros(size(Z,1),size(Z,2),k);
        for i =1:k
            dtz(:,:,i) = (-1)^k*diff(cat(3,Z(:,:,end-k+i:end),Z(:,:,1:i)),k,3);
        end

        Dtz = cat(3,dtz,Dtz);
    else
        Dtz = 0;
    end


    dtxy = 2*Dty + Dtz;
    dtxy = dtxy(:);
else
    %standard l1 minimization if order is 0
    dtxy = X(:)+Y(:)+Z(:);
end
    
    
    
    