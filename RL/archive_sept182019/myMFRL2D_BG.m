function [u,out] = myMFRL2D_BG(h,b,bg,opts)

% multi-frame Richardson-Lucy algorithm for deblurring the multi-frame data 
% b with the blurring kernels h 
% bg is the background
% opts contains options with fields:
% tol - tolerance
% iter - maximum iterations

% initialize...
if nargin<3, opts.iter = 100; end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'lambda'), opts.lambda = 0; end % regularization param
if ~isfield(opts,'order'), opts.order = 1; end % regularization order
if ~isfield(opts,'fast'), opts.fast = true; end % acceleration option
nF = size(b,3);
epsilon = 1e-10;
lam = opts.lambda;
order = opts.order;


hhat = fft2(h); % transform psf into Fourier domain
u = max(ifft2(sum(fft2(b-bg).*hhat,3))/nF,1e-1); % initial guess for u
up = u;
out.rel_chg = [];out.objF = [];
out.timer = [];
for i = 1:opts.iter
   tic;
   % acceleration -> iterate from new point, y
   if opts.fast, NestAlpha = (i-1)/(i+2);
   else, NestAlpha = 0;
   end
   y = max(u + NestAlpha*(u-up),1e-1);
   up = u;
   
   Hy = ifft2(hhat.*fft2(y)) + bg;
   bDHy = (b+epsilon)./(Hy+epsilon); % b/h*u
   if opts.lambda == 0
       % u(y) <- u(y)*(h(-y)*(b/h*u))
       u = real(y.*(ifft2(sum(conj(hhat).*fft2(bDHy),3)))/nF); 
   else
       gradR = lam*functionalDerHOTV(y,1e-1,order);
       % split gradR into positive and negative parts
       S1 = double(gradR>=0);
       S2 = abs(S1-1);
       u = y.*(ifft2(sum(conj(hhat).*fft2(bDHy),3))/nF-gradR.*S2)./(1+gradR.*S1);
       %  u = (u./(1+gradR)).*(ifftn(conj(hhat).*fftn(bDHu)));
       u = max(u,1e-1);
   end
   % check convergence
   out.rel_chg = [out.rel_chg;myrel(u,up)]; 
   out.timer = [out.timer;toc];
   if out.rel_chg(end)<tol, break;end

   % compute obj. functional
   % compute outside of timer since only needed for output metrics
   Hu = ifft2(hhat.*fft2(u)); % h*u
   out.objF = [out.objF;-sum(Hu(:)) + sum(b(:).*log(Hu(:)))-lam*computeTVnorm(u,order)];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));


function uTV = computeTVnorm(u,order)
if order == 0, uTV = sum(sum(abs(u))); return; end

ux = diff(u,order,2);
uy = diff(u,order,1);
uTV = sum(sum(sqrt(ux(1:end-order,:).^2 + uy(:,1:end-order).^2)));

