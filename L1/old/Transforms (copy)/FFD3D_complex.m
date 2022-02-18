function [D,Dt] = FFD3D_complex(k,p,q,r,theta)


% Written by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016


% finite difference operators for polynomial annihilation
% k is the order of the PA transform
VX = (exp(1i*2*pi*(0:q-1)/q)-1).^k;
VY = (exp(1i*2*pi*(0:p-1)/p)-1).^k;
VZ = (exp(1i*2*pi*(0:r-1)/r)-1).^k;

[VX,VY,VZ] = meshgrid(VX,VY,VZ);

D = @(U)ForwardD(U,VX,VY,VZ,theta);
Dt = @(X,Y,Z)Dive(X,Y,Z,VX,VY,VZ,theta);


% high order finite differences
function [X,Y,Z] = ForwardD(U,VX,VY,VZ,theta)
    U = U.*theta;
    [a,b,c] = size(U);

    X = fft(U,b,2);
    Y = fft(U,a,1);
    Z = fft(U,c,3);
    
    X = X.*VX;
    Y = Y.*VY;
    Z = Z.*VZ;

    X = ifft(X,b,2);
    Y = ifft(Y,a,1);
    Z = ifft(Z,c,3);

%transpose FD
function dtxy = Dive(X,Y,Z,VX,VY,VZ,theta)
[a,b,c] = size(X);
    
    X = fft(X,b,2);
    Y = fft(Y,a,1);
    Z = fft(Z,c,3);
    
    
    X = X.*conj(VX);
    Y = Y.*conj(VY);
    Z = Z.*conj(VZ);

    X = ifft(X,b,2);
    Y = ifft(Y,a,1);
    Z = ifft(Z,c,3);

    dtxy = (X + Y + Z).*conj(theta);
    dtxy = dtxy(:);

    
    
    
    