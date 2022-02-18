function [D,Dt] = FD3D_multiscale(k,p,q,r,N)


% finite difference operators for higher order TV
% k is the order of the transform
% levels is the number of scales used for the FD transforms
% recommended 3 levels
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 09/24/2016

x = linspace(-1,1,N);
A = fliplr(vander(x));
b = zeros(N,1); b(k+1)=1;
w = A'\b;

% rescale weights
w= k*w/abs(w(round(N/2)));

vx = fft(w,q);
vy = fft(w,p);
vz = fft(w,r);
[VX,VY,VZ] = meshgrid(vx,vy,vz);





D = @(U)D_Forward(U,VX,VY,VZ,p,q,r,k);
Dt = @(Uc)D_Adjoint(Uc,VX,VY,VZ,p,q,r,k);


% multiscale high order finite differences
function [dU] = D_Forward(U,VX,VY,VZ,p,q,r,k)

    U = reshape(U,p,q,r);
    
    % Transform data into frequency domain along each dimension
    % and allocate FD matrices for storage
    Fx = fft(U,q,2); X = zeros(p,q,r);
    Fy = fft(U,p,1); Y = X;
    Fz = fft(U,r,3); Z = X;
    
    % filtering for each level and dimension

    X(:,:,:) = Fx.*VX(:,:,:);
    Y(:,:,:) = Fy.*VY(:,:,:);
    Z(:,:,:) = Fz.*VZ(:,:,:);
    
    % transform back to real space
    X = ifft(X,q,2);
    Y = ifft(Y,p,1);
    Z = ifft(Z,r,3);
    
    % reshape data into 3 vectors
    dU = 2^(1-k)*[X(:),Y(:),Z(:)];
    




% transpose FD
function dtxy = D_Adjoint(dU,VX,VY,VZ,p,q,r,k)
    
    X = reshape(dU(:,1),p,q,r);
    Y = reshape(dU(:,2),p,q,r);
    Z = reshape(dU(:,3),p,q,r);
    
    % transform data into frequency domain along each dimension
    X = fft(X,q,2);
    Y = fft(Y,p,1);
    Z = fft(Z,r,3);
    
    % conjugate filtering for each level and dimension
    X(:,:,:) = X(:,:,:).*conj(VX);
    Y(:,:,:) = Y(:,:,:).*conj(VY);
    Z(:,:,:) = Z(:,:,:).*conj(VZ);
    % transform filtered data back to real space
    X = ifft(X,q,2);
    Y = ifft(Y,p,1);
    Z = ifft(Z,r,3);
    
    dtxy = 2^(1-k)*(X(:) + Y(:) + Z(:));

        

    
    
    
    