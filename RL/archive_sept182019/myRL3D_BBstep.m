function [u,out] = myRL3D_BBstep(h,b,n,opts)

% HOTV-MAP formulation for Poisson image deblurring
% this code uses gradient decent with BB-step

% blurring kernel (h), where the image size is n (pxqxr) 
% b - blurred image
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
epsilon = 1e-1;
lam = opts.lambda;
order = opts.order;

hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
u = ifft2(fft2(b).*hhat); % initial guess for u
Hu = ifftn(hhat.*fftn(u)); % h*u
bDHu = (b+epsilon)./(Hu+epsilon); % b/h*u, "ratio"
g = ifftn(conj(hhat).*fftn(1-bDHu));  % gradient "vector" (image)

% R-L algorithm in first iteration    
up = u;
u = u.*(ifftn(conj(hhat).*fftn(bDHu))); % u(y) <- u(y)*(h(-y)*(b/h*u))
Hu = ifftn(hhat.*fftn(u)); % h*u
out.objF = -sum(Hu(:)) + sum(b(:).*log(Hu(:)))- lam*computeTVnorm(u,order);
out.rel_chg = []; out.cnt = [];out.tau = [];
out.timer = [];
for i = 1:opts.iter 
   tic;
   bDHu = (b+epsilon)./(Hu+epsilon); % b/h*u, "ratio"
   % gradient "vector" (image)
   gp = g; % save previous gradient image
   g = ifftn(conj(hhat).*fftn(1-bDHu)) + lam*functionalDerHOTV(u,epsilon,order);  
   % BB-step gradient descent    
   uup = u(:) - up(:);
   tau = (uup'*uup)/(uup'*col(g-gp));   % BB-step      
   up = u; % save previous solution for next tau
   u = max(u - tau*g,1e-10); % descent
   Hu = ifftn(hhat.*fftn(u)); % h*u     
   
   % check convergence and compute obj. functional
   out.tau = [out.tau;tau];
   out.rel_chg = [out.rel_chg;myrel(u,up)];
   out.timer = [out.timer;toc];
   out.objF = [out.objF;sum(-Hu(:) + b(:).*log(Hu(:))) - lam*computeTVnorm(u,order)];
   if out.rel_chg(end)<tol, break;end
   
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));

function uTV = computeTVnorm(u,order)
if order == 0, uTV = sum(sum(abs(u))); return; end

ux = diff(u,order,2);
uy = diff(u,order,1);
uTV = sum(sum(sqrt(ux(1:end-order,:).^2 + uy(:,1:end-order).^2)));
