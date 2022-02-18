function [u,out] = myRL3D_LineSearch(h,b,n,opts)

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
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
epsilon = 1e-10;
% u = ones(p,q)*100;

% iterate 
hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
u = ifft2(fft2(b).*hhat); % initial guess for u
Hu = ifftn(hhat.*fftn(u)); % h*u
out.rel_chg = [];out.objF = [];out.tau = [];
for i = 1:opts.iter
   bDHu = (b+epsilon)./(Hu+epsilon); % b/h*u
   up = u;
   u = u.*(ifftn(conj(hhat).*fftn(bDHu))); % u(y) <- u(y)*(h(-y)*(b/h*u))
   g = u-up;  % form the "gradient" vector
   
   % line search here
   Hg = ifftn(fftn(g).*hhat);
   [tau,objF] = basicPoisLineSearch(Hu,Hg,b,2);
   u = max(up + tau*g,0);    
   Hu = ifftn(hhat.*fftn(u)); % h*u   
   
   % check convergence
   out.rel_chg = [out.rel_chg;myrel(u,up)]; 
   out.tau = [out.tau;tau];
   if out.rel_chg(end)<tol, break;end
   
   % compute obj. functional
   out.objF = [out.objF;objF];% -sum(Hu(:)) + sum(b(:).*log(Hu(:)))];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));