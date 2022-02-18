function B = getTikhonovDenoise1D(k,p,q)

VY = (exp(1i*2*pi*(0:p-1)/p)-1).^k;
VY = (VY.*conj(VY))';
B = @(x,lambda)localTikDenoise(x,lambda,VY);

function x = localTikDenoise(x,lambda,VY)

Fx = fft(x);
x = real(ifft(Fx./(1 + lambda*VY)));