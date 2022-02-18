function [x,out,xr] = myRLTik(h,b,n,opts)


if nargin<4
    opts.iter = 100;
end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
if ~isfield(opts,'order'), opts.order = 2; end
if ~isfield(opts,'levels'), opts.levels = 1; end
if ~isfield(opts,'disp'), opts.disp = 0; end
p = n(1); q = n(2); r = n(3);

epsilon = 1e-10;
lambda = 1e-5;
levels = opts.levels;
k = opts.order;
mu = 1e-2;

% store eigenvalues of regularization operator 
vx = zeros(levels,q); vy = zeros(levels,p);
for i = 1:levels
    vx(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,q-1,q)/q).^(2*k+2))./...
            (sin(pi*linspace(0,q-1,q)/q).^2)*(2^(2-k-i)/levels)^2;
    vy(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,p-1,p)/p).^(2*k+2))./...
        (sin(pi*linspace(0,p-1,p)/p).^2)*(2^(2-k-i)/levels)^2;
end
vx(:,1) = 0; vy(:,1) = 0;
vx = sum(vx,1); vy = sum(vy,1);
V = vy' + vx;


x = b+1;xr = x;
hhat = fftn(h,[p,q,r]);
out.rel_error = [];
for i = 1:opts.iter
   Fx = fft2(x);
   Hx = ifftn(hhat.*Fx); % convolution of x with PSF
   gRx = mu*ifft2(V.*Fx); % gradient of regularization term 
   mask1 = double(gRx>=0);
   mask2 = abs(mask1-1);
   bDHx = (b+epsilon)./(Hx+epsilon);
   xp = x;
   x = x.*(ifftn(conj(hhat).*fftn(bDHx))-gRx.*mask2)./(1 + gRx.*mask1);
   x = max(x,epsilon);
   out.rel_error = [out.rel_error;myrel(x,xp)];
   if out.rel_error(end)<tol, break;end
   if opts.disp
       figure(75);
       subplot(2,2,1);imagesc(real(x));title(i);colorbar;
       subplot(2,2,2);imagesc(imag(x));title('imag');colorbar;
       subplot(2,2,3);imagesc(V);title('HPF');colorbar;
       subplot(2,2,4);imagesc(1./(1+V));title('LPF (division)');colorbar;
   end
end
out.iters = i;