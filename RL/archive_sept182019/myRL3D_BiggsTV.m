function [u,out] = myRL3D_BiggsTV(h,b,n,opts)

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
lam = opts.lambda;

% iterate 
hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
u = ifft2(fft2(b).*hhat); % initial guess for u
up = u;
alpha = 0;
g = zeros(p*q*r,1);
out.rel_chg = [];out.objF = [];out.alpha = [];out.tau = [];
Hu = ifftn(hhat.*fftn(u)); % h*u
for i = 1:opts.iter
   v = max(u + alpha*(u-up),1e-9);
   Hv = ifftn(hhat.*fftn(v)); % h*v
   bDHv = (b+epsilon)./(Hv+epsilon); % b/h*u
   up = u;
   gp = g;
   if opts.lambda == 0
      u = v.*(ifftn(conj(hhat).*fftn(bDHv))); % u(y) <- u(y)*(h(-y)*(b/h*u))
   else
      TVterm = myTVEulerLagrange(u,1e-1);
      TVterm = lam*TVterm;
      % algorithm is slightly more stable by splitting TVterm into positive
      % and negative parts
      S1 = double(TVterm>=0);
      S2 = abs(S1-1);
      u = max(u.*(ifftn(conj(hhat).*fftn(bDHv))-TVterm.*S2)./(1+TVterm.*S1),1e-3);
      gg = u-up;  % form the "gradient" vector
   
       % line search here
       % u(tau) = up + tau*(u-up) "=" up - tau*g
       Hg = ifftn(fftn(gg).*hhat); 
       [tau,objF] = local_TVPoisLineSearch(Hu,Hg,up,gg,b,lam,2);
       out.tau = [out.tau;tau];
      %  tau = .1;
       u = max(up + tau*gg,0);      
   end
   g = u(:)-v(:);
   % alpha = max(min((g'*gp)/(g'*g),.999),0);
   alpha = (i-1)/(i+2);
   % check convergence.
   out.rel_chg = [out.rel_chg;myrel(u,up)];
   out.alpha = [out.alpha;alpha];
   if out.rel_chg(end)<tol, break;end
   Hu = ifftn(hhat.*fftn(u)); % h*u
    
 %  figure(13);imagesc(u);title(i);pause;
   
   % compute obj. functional
   out.objF = [out.objF;-sum(Hu(:)) + sum(b(:).*log(Hu(:)))-computeTVnorm(u,lam)];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));


function uTV = computeTVnorm(u,lambda)

ux = diff(u,1,2);
uy = diff(u,1,1);
uTV = lambda*sum(sum(sqrt(ux(1:end-1,:).^2 + uy(:,1:end-1).^2)));


function [tau,fval]= local_TVPoisLineSearch(Hu,Hg,up,g,b,lam,gam)

% line search for Poisson log likelihood
% searching along step tau by
%      Unew = Up + tau(U-Up)
% return:
% tau - good estimate for step length
% fval - function value at that location

% objective function value is
% sum(-Hu - Hg + b.*log(Hu + Hg)) - lam*TV(u)
if nargin<5, gam = 2; end

sumHg = sum(Hg(:));
sumHu = sum(Hu(:));      
objFold = -sum(Hu(:)) + sum(b(:).*log(Hu(:))) - computeTVnorm(up,lam);
objFnew = -(sumHu+sumHg) + sum(b(:).*log(Hu(:) + Hg(:))) - computeTVnorm(up+g,lam);
tau = 1;
if objFnew>objFold % forward line search
    while objFnew>objFold & imag(sum(objFnew(:)))==0
       tau = tau*gam;
       objFold = objFnew;
       objFnew = -(sumHu+tau*sumHg) + sum(b(:).*log(Hu(:) + tau*Hg(:)))...
           - computeTVnorm(up+tau*g,lam);
    end
    tau = tau/gam;
else % backtracking
    while objFnew<objFold & imag(sum(objFnew(:)))==0
       tau = tau/gam;
       objFnew = -(sumHu+tau*sumHg) + sum(b(:).*log(Hu(:) + tau*Hg(:)))...
           - computeTVnorm(up+tau*g,lam);        
    end
end

fval = objFold;
