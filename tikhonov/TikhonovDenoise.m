function x = TikhonovDenoise(x,k,mu)

[p,q,r] = size(x);
if r~=1, error('set up for 2D image denoising only'); end

Fx = fft2(x);
VX = (exp(1i*2*pi*(0:q-1)/q)-1).^k;
VY = (exp(1i*2*pi*(0:p-1)/p)-1).^k;
[VX,VY] = meshgrid(VX,VY);

x = real(ifft2(Fx./(mu + VX + VY)));