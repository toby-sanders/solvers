function [u,out] = myRL3D(h,b,n,opts)

% standard Richardson-Lucy algorithm for deblurring the data (b) with the
% blurring kernel (h), where the image size is n (pxqxr) 
% opts contains options with fields:
% tol - tolerance
% iter - maximum iterations

% initialize...
if nargin<4, opts.iter = 100; end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'lambda'), opts.lambda = 0; end
if ~isfield(opts,'order'), opts.order = 1; end
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
epsilon = 1e-10;
lam = opts.lambda;
order = opts.order;

% initial guess for u
% u = ones(p,q)*100;

% iterate 
hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
u = ifft2(fft2(b).*hhat); % initial guess for u
out.rel_chg = [];out.objF = [];
out.timer = [];
Hu = ifftn(hhat.*fftn(u)); % h*u
for i = 1:opts.iter
   tic;
   bDHu = (b+epsilon)./(Hu+epsilon); % b/h*u
   up = u;
   if opts.lambda == 0
       u = u.*(ifftn(conj(hhat).*fftn(bDHu))); % u(y) <- u(y)*(h(-y)*(b/h*u))
   else
      gradR = lam*functionalDerHOTV(u,1e-1,order);
      % algorithm is slightly more stable by splitting gradR into positive
      % and negative parts
      S1 = double(gradR>=0);
      S2 = abs(S1-1);
      u = u.*(ifftn(conj(hhat).*fftn(bDHu))-gradR.*S2)./(1+gradR.*S1);
     %  u = (u./(1+gradR)).*(ifftn(conj(hhat).*fftn(bDHu)));
       u = max(u,1e-1);
   end
   % check convergence
   out.rel_chg = [out.rel_chg;myrel(u,up)];
   Hu = ifftn(hhat.*fftn(u)); % h*u
   out.timer = [out.timer;toc];
   if out.rel_chg(end)<tol, break;end

   % compute obj. functional
   out.objF = [out.objF;-sum(Hu(:)) + sum(b(:).*log(Hu(:)))-lam*computeTVnorm(u,order)];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));


function uTV = computeTVnorm(u,order)
if order == 0, uTV = sum(sum(abs(u))); return; end

ux = diff(u,order,2);
uy = diff(u,order,1);
uTV = sum(sum(sqrt(ux(1:end-order,:).^2 + uy(:,1:end-order).^2)));

