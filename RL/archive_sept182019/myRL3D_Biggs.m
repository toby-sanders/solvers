function [u,out] = myRL3D_Biggs(h,b,n,opts)

% this version implements Biggs accelerated method (1997), which is closely
% related to Nesterov's accelerated gradient descent (1983).  I have also 
% modified the selection of "alpha" to match Nesterov's approach.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The accelerated algorithm takes the form
% v = u + alpha(u - up)
% u = psi(v),
% where psi is the Richardson-Lucy algorithm/iteration.

% Richardson-Lucy algorithm for deblurring the data (b) with the
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
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
epsilon = 1e-10;

% iterate 
hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
u = ifft2(fft2(b).*hhat); % initial guess for u
up = u;
alpha = 0;
g = zeros(p*q*r,1);
out.rel_chg = [];out.objF = [];out.alpha = [];
Hu = ifftn(hhat.*fftn(u)); % h*u
out.timer = [];
for i = 1:opts.iter
   tic;
   % v is determined from previous two iterates
   v = max(u + alpha*(u-up),1e-9);
   
   % UPDATE u from v
   % u(y) <- v(y)*(h(-y)*(b/h*v))
   Hv = ifftn(hhat.*fftn(v)); % h*v
   bDHv = (b+epsilon)./(Hv+epsilon); % b/h*u
   up = u;
   u = v.*(ifftn(conj(hhat).*fftn(bDHv))); 
   if i==1 % line search in first iteration, though not necessary
      Hg = ifftn(fftn(u-up).*hhat);
      [tau,objF] = basicPoisLineSearch(Hu,Hg,b,2);
      u = max(up + tau*(u-up),0); 
   end
   g = u(:)-v(:);
   % alpha = max(min((g'*gp)/(g'*g),.999),0);
   alpha = (i-1)/(i+2);

   % check convergence.
   out.rel_chg = [out.rel_chg;myrel(u,up)];
   out.alpha = [out.alpha;alpha];
   if out.rel_chg(end)<tol, break;end
   Hu = ifftn(hhat.*fftn(u)); % h*u
   out.timer = [out.timer;toc];
   % compute obj. functional
   out.objF = [out.objF;-sum(Hu(:)) + sum(b(:).*log(Hu(:)))];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));

