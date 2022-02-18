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
if ~isfield(opts,'lambda'), opts.lambda = 0; end
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
epsilon = 1e-10;
% initial guess for u
u = ones(p,q)*100;
lam = opts.lambda;

% iterate 
hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
Hu = ifftn(hhat.*fftn(u)); % h*u
out.rel_chg = [];out.objF = [];out.tau = [];
out.TVnorm = computeTVnorm(u,lam);
for i = 1:opts.iter
   bDHu = (b+epsilon)./(Hu+epsilon); % b/h*u
   up = u;
   TVterm = -myTVEulerLagrange(u,1e-1);
   TVterm = opts.lambda*TVterm;
   % algorithm is slightly more stable by splitting TVterm into positive
   % and negative parts
   S1 = double(TVterm>=0);
   S2 = abs(S1-1);
   u = u.*(ifftn(conj(hhat).*fftn(bDHu))-TVterm.*S2)./(1+TVterm.*S1);
  %  u = u.*(ifftn(conj(hhat).*fftn(bDHu))); % u(y) <- u(y)*(h(-y)*(b/h*u))
   g = u-up;  % form the "gradient" vector
   
   % line search here
   % u(tau) = up + tau*(u-up) "=" up - tau*g
   % objective function value is
   % sum(-Hu - Hg + b.*log(Hu + Hg))
   % should be slightly more efficient to us Hup and Hu in future 
   Hg = ifftn(fftn(g).*hhat); 
   sumHg = sum(Hg(:));
   sumHu = sum(Hu(:));      
   objFold = -sum(Hu(:)) + sum(b(:).*log(Hu(:)));% - computeTVnorm(up,lam);
   objFnew = -(sumHu+sumHg) + sum(b(:).*log(Hu(:) + Hg(:)));% - computeTVnorm(u,lam);
   tau = 1;
   while objFnew>objFold & imag(sum(objFnew(:)))==0 & tau<16
       objFold = objFnew;
       tau = 2*tau;
       objFnew = -(sumHu+tau*sumHg) + sum(b(:).*log(Hu(:) + tau*Hg(:)));
       % ...
        %   - computeTVnorm(up+tau*g,lam);
   end
   if tau<20, tau = tau/2; end
   out.tau = [out.tau;tau];
   u = max(up + tau*g,0);
       
   Hu = ifftn(hhat.*fftn(u)); % h*u
      
   % check convergence
   out.rel_chg = [out.rel_chg;myrel(u,up)];
   if out.rel_chg(end)<tol, break;end
   
   % compute obj. functional
   out.objF = [out.objF;-sum(Hu(:)) + sum(b(:).*log(Hu(:)))];%- computeTVnorm(u,lam)];
   out.TVnorm = [out.TVnorm;computeTVnorm(u,lam)];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));

function uTV = computeTVnorm(u,lambda)

ux = diff(u,1,2);
uy = diff(u,1,1);
uTV = lambda*sum(sum(sqrt(ux(1:end-1,:).^2 + uy(:,1:end-1).^2)));






