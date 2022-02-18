function [u,out] = myLucy(h,b,opts)

% multi-frame Richardson-Lucy algorithm for deblurring the multi-frame data 
% b with the blurring kernels h 
% opts contains options with fields:
% tol - tolerance
% iter - maximum iterations
% bg - background image 
% fast - Nesterov acceleration option (default true)
% mask - sensitivity/bad pixel mask (default all ones)


% initialize...
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'lambda'), opts.lambda = 0; end % regularization param
if ~isfield(opts,'order'), opts.order = 1; end % regularization order
if ~isfield(opts,'levels'), opts.levels = 1; end % number of scaling factors
if ~isfield(opts,'fast'), opts.fast = true; end % acceleration option
if ~isfield(opts,'mask'), mask = ones(size(b,1),size(b,2));
else, mask = opts.mask; end
if ~isfield(opts,'bg'), bg = zeros(size(b,1),size(b,2));
else, bg = opts.bg; end
if ~isfield(opts,'superRes'), superRes = false;
else, superRes = opts.superRes; end
if ~isfield(opts,'iter')
    if opts.fast, opts.iter = 50;
    else, opts.iter = 200;
    end
end
nF = size(b,3); % number of frames
epsilon = 1e-10;
lam = opts.lambda;


if superRes
    if size(h,1)~=3*size(b,1) % h may have been input in superRes form
        h = imresize(h,3*[size(h,1),size(h,2)])/9;
    end
    tmp = zeros(3*size(b,1),3*size(b,2),size(b,3));
    tmp(2:3:end,2:3:end,:) = b;
    b = tmp;
    if size(bg,3)==1, tmp(2:3:end,2:3:end,:) = repmat(bg,1,1,nF);
    else, tmp(2:3:end,2:3:end,:) = bg;
    end
    bg = tmp;
    tmp = tmp(:,:,1);
    tmp(2:3:end,2:3:end) = mask;
    mask = tmp;
   
    ghat = zeros(size(b,1),size(b,2));
    ghat([1:2,end],[1:2,end]) = 1/9;
    ghat = fft2(ghat); % 9 pixel averaging kernel
end

% get preallocated MHOTV operators
D_MHOTV = getMHOTVfunctionDer(opts.order,opts.levels,size(b,1),size(b,2));
    


hhat = fft2(h); % transform psf into Fourier domain
if superRes, hhat = hhat.*ghat; end
u = max(ifft2(sum(fft2(b-bg).*hhat.*hhat.*hhat,3))/nF,1e-1); % initial guess for u
% u(:) = mean(u(:));
up = u;
denomObj = real(sum(ifft2(fft2(mask).*conj(hhat)),3));
out.rel_chg = [];out.objF = []; out.logLikelihood = [];
out.timer = [];
sumB = - sum(b(:).*log(b(:))) + sum(b(:));
normR = 0;
for i = 1:opts.iter
   tic;
   % acceleration -> iterate from new point, y
   if opts.fast, NestAlpha = (i-1)/(i+2);
   else, NestAlpha = 0;
   end
   y = max(u + NestAlpha*(u-up),1e-1);
   up = u;
   
   Hy = mask.*ifft2(hhat.*fft2(y)) + bg;
   bDHy = (b+epsilon)./(Hy+epsilon); % b/h*u
   if lam == 0
       % u(x) <- u(x)*(h(-x)*(b/h*u))
       u = real(y.*(ifft2(sum(conj(hhat).*fft2(bDHy.*mask),3)))./denomObj); 
   else
       % gradR = lam*functionalDerHOTV(y,1e-1,order);
       % gradR = lam*MHOTVfunctionDer(y,opts.order,opts.levels);
       [gradR,normR] = D_MHOTV(y);
       % split gradR into positive and negative parts
       S1 = double(gradR>=0);
       S2 = abs(S1-1);
      % u = y.*(ifft2(sum(conj(hhat).*fft2(bDHy.*mask),3))./denomObj-lam*gradR.*S2)./(1+lam*gradR.*S1);
       u = y.*ifft2(sum(conj(hhat).*fft2(bDHy.*mask),3))./denomObj./(1+lam*gradR);
       u = max(real(u),1e-1);
   end

   % check convergence
   out.rel_chg = [out.rel_chg;myrel(u,up)]; 
   out.timer = [out.timer;toc];
   if out.rel_chg(end)<tol, break;end

   % compute obj. functional (outside of timer since only for output stats)
   if ~superRes
       out.logLikelihood = [out.logLikelihood; ...
           -sum(Hy(:)) + sum(b(:).*log(Hy(:))) + sumB];
           out.objF = [out.objF;out.logLikelihood(end)-lam*sum(normR(:))];
   else
   out.objF = [out.objF;real(-sum(col(Hy(:))) -lam*sum(normR(:))...
       + sum(col(b(2:3:end,2:3:end,:).*log(Hy(2:3:end,2:3:end,:)))))];
   end
end
out.iters = i;
% out.logLikelihood = out.objF - sum(b(:).*log(b(:))) + sum(b(:));
out.objF = real(out.objF);
out.hhat = hhat;

% function uTV = computeTVnorm(u,order)
% if order == 0, uTV = sum(sum(abs(u))); return; end
% 
% ux = diff(u,order,2);
% uy = diff(u,order,1);
% uTV = sum(sum(sqrt(ux(1:end-order,:).^2 + uy(:,1:end-order).^2)));

