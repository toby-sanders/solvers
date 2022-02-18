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
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
epsilon = 1e-10;
lam = opts.lambda;

% initial guess for u
% u = ones(p,q)*100;

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
   if opts.lambda == 0 || i<2
        grad = ifftn(conj(hhat).*fftn(1-bDHv)); % gradient "vector" (image)
        grad = grad + lam*myTVEulerLagrange(v,epsilon);
      u = v.*(ifftn(conj(hhat).*fftn(bDHv))); % u(y) <- u(y)*(h(-y)*(b/h*u))
      objF=    -sum(Hu(:)) + sum(b(:).*log(Hu(:)))-lam*computeTVnorm(u);
   else
       % try a BB-step with backtracking
        grad = ifftn(conj(hhat).*fftn(1-bDHv)); % gradient "vector" (image)
        grad = grad + lam*myTVEulerLagrange(v,epsilon);
        
        tau = (uup'*uup)/(uup'*col(grad-gradp));   % BB-step
        % gradient decent, save previous U and convolved U for backtracking
        up = u; Hup = Hu;
        u = max(u - tau*grad,1e-10); % decent
        Hu = ifftn(hhat.*fftn(u)); % h*u
      
        % objFp = objF; % previous objective function value
        objF = -sum(Hu(:)) + sum(b(:).*log(Hu(:))) - lam*computeTVnorm(u); % new obj. value
        
   
       % line search here
       % u(tau) = up + tau*(u-up) "=" up - tau*g
%        Hg = ifftn(fftn(gg).*hhat); 
%        [tau,objF] = TVPoisLineSearch(Hu,Hg,up,gg,b,lam,2);
       out.tau = [out.tau;tau];
      %  tau = .1;
       % u = max(up + tau*gg,0);      
   end
   uup = u(:)-up(:);
   gradp = grad;
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
   out.objF = [out.objF;objF];% -sum(Hu(:)) + sum(b(:).*log(Hu(:)))-lam*computeTVnorm(u)];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));


function uTV = computeTVnorm(u)

ux = diff(u,1,2);
uy = diff(u,1,1);
uTV = sum(sum(sqrt(ux(1:end-1,:).^2 + uy(:,1:end-1).^2)));

