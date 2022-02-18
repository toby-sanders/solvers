function [D,Dt] = FD3D_multiscale(k,p,q,r,levels)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016


% finite difference operators for polynomial annihilation
% k is the order of the PA transform
% levels is the number of scales used for the FD transforms
% recommended 3 levels


vy = zeros(p,1); vx = zeros(q,1); vz = zeros(r,1);
if round(k)==k
    for ii = 0:k
        if k < p
            vy(ii+1) = (-1)^ii*nchoosek(k,ii);
        end
        if k < q
            vx(ii+1) = (-1)^ii*nchoosek(k,ii);
        end
        if k < r
            vz(ii+1) = (-1)^ii*nchoosek(k,ii);
        end
    end
else
    vx = ifft((exp(1i*2*pi*(0:q-1)/q)-1).^k)';
    vy = ifft((exp(1i*2*pi*(0:p-1)/p)-1).^k)';
    vz = ifft((exp(1i*2*pi*(0:r-1)/r)-1).^k)';
end
VX = zeros(p,q,r,levels); VY = VX; VZ = VX;


[VX(:,:,:,1),VY(:,:,:,1),VZ(:,:,:,1)] = meshgrid(fft(vx),fft(vy),fft(vz));

for ii = 1:levels
    %vy = imresize(vy,[2*p,1],'nearest');vy = vy(1:p);
    %vx = imresize(vx,[2*q,1],'nearest');vx = vx(1:q);
    %vz = imresize(vz,[2*r,1],'nearest');vz = vz(1:r);
    %[VX(:,:,:,ii),VY(:,:,:,ii),VZ(:,:,:,ii)] = ...
    %    meshgrid(fft(vx)*2^(1-ii),fft(vy)*2^(1-ii),fft(vz)*2^(1-ii));
    vx = [0,2^(1-ii)*((exp(-1i*2*pi*(1:q-1)*2^(ii-1)/q)-1).^(k+1))./(exp(-1i*2*pi*(1:q-1)/q)-1)];
    vy = [0,2^(1-ii)*((exp(-1i*2*pi*(1:p-1)*2^(ii-1)/p)-1).^(k+1))./(exp(-1i*2*pi*(1:p-1)/p)-1)];
    vz = [0,2^(1-ii)*((exp(-1i*2*pi*(1:r-1)*2^(ii-1)/r)-1).^(k+1))./(exp(-1i*2*pi*(1:r-1)/r)-1)];
    [VX(:,:,:,ii),VY(:,:,:,ii),VZ(:,:,:,ii)] = meshgrid(vx,vy,vz);
end
    
    
    



D = @(U)D_Forward(U,VX,VY,VZ,p,q,r,levels);
Dt = @(Uc)D_Adjoint(Uc,VX,VY,VZ,p,q,r,levels);


% multiscale high order finite differences
function [dU] = D_Forward(U,VX,VY,VZ,p,q,r,levels)
    
    U = reshape(U,p,q,r);
    
    % Transform data into frequency domain along each dimension
    % and allocate FD matrices for storage
    Fx = fft(U,q,2); X = zeros(p,q,r,levels);
    Fy = fft(U,p,1); Y = X;
    Fz = fft(U,r,3); Z = X;
    
    % filtering for each level and dimension
    for i = 1:levels
        X(:,:,:,i) = Fx.*VX(:,:,:,i);
        Y(:,:,:,i) = Fy.*VY(:,:,:,i);
        Z(:,:,:,i) = Fz.*VZ(:,:,:,i);
    end
    
    % transform back to real space
    X = ifft(X,q,2);
    Y = ifft(Y,p,1);
    Z = ifft(Z,r,3);
    
    % reshape data into 3 vectors
    dU = [X(:),Y(:),Z(:)];
    




% transpose FD
function dtxy = D_Adjoint(dU,VX,VY,VZ,p,q,r,levels)
    
    X = reshape(dU(:,1),p,q,r,levels);
    Y = reshape(dU(:,2),p,q,r,levels);
    Z = reshape(dU(:,3),p,q,r,levels);
    
    % transform data into frequency domain along each dimension
    X = fft(X,q,2);
    Y = fft(Y,p,1);
    Z = fft(Z,r,3);
    
    % conjugate filtering for each level and dimension
    for i = 1:levels
        X(:,:,:,i) = X(:,:,:,i).*conj(VX(:,:,:,i));
        Y(:,:,:,i) = Y(:,:,:,i).*conj(VY(:,:,:,i));
        Z(:,:,:,i) = Z(:,:,:,i).*conj(VZ(:,:,:,i));
    end
    
    % transform filter data back to real space
    X = ifft(X,q,2);
    Y = ifft(Y,p,1);
    Z = ifft(Z,r,3);
    
    % finish transpose operation by appropriate summing
    X = sum(X,4); Y = sum(Y,4); Z = sum(Z,4);
    
    dtxy = X(:) + Y(:) + Z(:);

        

    
    
    
    