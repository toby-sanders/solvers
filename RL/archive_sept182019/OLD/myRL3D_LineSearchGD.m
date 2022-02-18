function [u,out] = myRL3D_BBstep(h,b,n,opts)

% ML formulation for Poisson image deblurring
% this code uses gradient decent with BB-step and backtracking

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
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
epsilon = 1e-1;
gam = .5; % shrinkage factor in the backtracking

% u = ones(p,q)*10; 

% iterate 
hhat = fftn(h,[p,q,r]); % transform psf into Fourier domain
u = ifft2(fft2(b).*hhat); % initial guess for u
out.rel_chg = [];out.objF = []; out.cnt = [];out.flg  =[];
Hu = ifftn(hhat.*fftn(u)); % h*u
for i = 1:opts.iter 
   bDHu = (b+epsilon)./(Hu+epsilon); % b/h*u, "ratio"
   if i<10 || i>opts.iter*.8
       % R-L algorithm in first and last iterations
        up = u;
        u = u.*(ifftn(conj(hhat).*fftn(bDHu))); % u(y) <- u(y)*(h(-y)*(b/h*u))
        g = ifftn(conj(hhat).*fftn(1-bDHu));
        Hu = ifftn(hhat.*fftn(u)); % h*u
        objF = -sum(Hu(:)) + sum(b(:).*log(Hu(:)));
   else
        % try a BB-step with backtracking
        g = ifftn(conj(hhat).*fftn(1-bDHu)); % gradient "vector" (image)
        tau = (uup'*uup)/(uup'*col(g-gp));   % BB-step
        % gradient decent, save previous U and convolved U for backtracking
        up = u; Hup = Hu;
        u = max(u - tau*g,1e-10); % decent
        Hu = ifftn(hhat.*fftn(u)); % h*u
      
        objFp = objF; % previous objective function value
        objF = -sum(Hu(:)) + sum(b(:).*log(Hu(:))); % new obj. value
        
        % backtracking, currently a simple half step back until objective
        % function increases.  In future can implement more involved
        % wolfe/armijo conditions
        cnt = 0;
        alpha = 1;
        if objFp>objF
            flg = 0;
            while objF<objFp 
               cnt = cnt+1;
               if cnt ==5  % give up and implement R-L
                   % u(y) <- u(y)*(h(-y)*(b/h*u))
                   u = up.*(ifftn(conj(hhat).*fftn(bDHu))); 
                   break;
               end
               alpha = alpha*gam; % shrink step length
               % recompute objective function at new estimate
               objF = sum(-alpha*Hu(:) + -(1-alpha)*Hup(:)) ...
                   + sum(b(:).*log(alpha*Hu(:) + (1-alpha)*Hup(:)));
            end    
            if cnt<5
                u = alpha*u + (1-alpha)*up;
            end
            % recompute h*u since backtracking was implemented
            Hu = ifftn(hhat.*fftn(u));
        else
            flg = 1;
            while objF>objFp
                cnt = cnt+1;
                objFp = objF;
                alpha = alpha/gam;
                % recompute objective function at new estimate
                objF = sum(-alpha*Hu(:) + -(1-alpha)*Hup(:)) ...
                   + sum(b(:).*log(alpha*Hu(:) + (1-alpha)*Hup(:)));
            end
            alpha = alpha*gam;
            u = alpha*u + (1-alpha)*up;
            % recompute h*u since backtracking was implemented
            Hu = ifftn(hhat.*fftn(u));
        end
        out.flg = [out.flg;flg];
        out.cnt = [out.cnt;cnt];
   end

%    figure(90);
%    subplot(2,2,1);imagesc(u);colorbar;title(i);
%    subplot(2,2,2);imagesc(g);colorbar;title('gradient');
%    subplot(2,2,3);semilogy(out.rel_chg);
%    subplot(2,2,4);semilogy(out.objF);
%    end

   % save u-up and previous gradient
   uup = u(:)-up(:);
   gp = g;
   % check convergence
   out.rel_chg = [out.rel_chg;myrel(u,up)];
   if out.rel_chg(end)<tol, break;end
   
   % compute obj. functional
   out.objF = [out.objF;-sum(Hu(:)) + sum(b(:).*log(Hu(:)))];
end
out.iters = i;
out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));










