function B = getTikhonovDenoise(k,p,q,r)
% this function returns an operator B, which will denoise an image of
% dimension pxq with an order k Tikhonov model.  This is done simply by
% applying a low pass filter.  The denoising model is given by
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       min_x ||x-y||^2 + lambda || Tx ||^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VX = (exp(1i*2*pi*(0:q-1)/q)-1).^k;
VY = (exp(1i*2*pi*(0:p-1)/p)-1).^k;
VZ = (exp(1i*2*pi*(0:r-1)/r)-1).^k;
[VX,VY,VZ] = meshgrid(VX,VY,VZ);
V = (VX.*conj(VX) + VY.*conj(VY) + VZ.*conj(VZ))/2^k;
B = @(x,lambda)localTikDenoise(x,lambda,V);

function x = localTikDenoise(x,lambda,V)

Fx = fft2(x);
x = real(ifft2(Fx./(1 + lambda*V)));