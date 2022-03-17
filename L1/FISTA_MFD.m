function [x,out] = FISTA(hhat,bhat,opts)

% (fast) iterative shrinkage thresholding algorithm (FISTA)
% this example is for deconvolution
% h is the PSF
% b is the blurry image (signal)
% opts contains options
% minimize mu/2 || h*x - b||_2^2 + ||x||_1 

[p,q,nF] = size(bhat);
if size(hhat,1)~=p || size(hhat,2)~=q || size(hhat,3)~=nF
    error('PSF and image data not compatible sizes');
end

if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'tol'), opts.tol = 1e-4; end
if ~isfield(opts,'mu')
    opts.mu = 10; % default mu
    warning('mu parameter was not set');
end
if ~isfield(opts,'fast'), opts.fast = true; end
if ~isfield(opts,'nonneg'), opts.nonneg = false;end



if ~isfield(opts,'init')
    x = rand(p,q);
else
    x = opts.init;
end

xp = x;
hhat2 = sum(hhat.*conj(hhat),3);
Atb = sum(ifft2(conj(hhat).*bhat),3);
out.rel_chg = [];
mu = opts.mu;
out.objF = [];
tau = 1/max(abs(hhat2(:)));
for i = 1:opts.iter
    if opts.fast, NestAlpha = (i-1)/(i+2);
    else, NestAlpha = 0;
    end
    
    y = x + NestAlpha*(x-xp);
    
    xp = x;
    x = y - real(tau*(ifft2(hhat2.*fft2(y))-Atb));
    x = max(abs(x)-1/mu,0).*sign(x);
    if opts.nonneg, x = max(x,0); end
    
    Fx = fft2(x);
    Axb = ifft2(hhat.*Fx-bhat);
    
    
    out.objF = [out.objF;mu/2*sum(col((Axb).*conj(Axb))) + sum(abs(x(:)))];
    out.rel_chg = [out.rel_chg;myrel(x,xp)];
    if out.rel_chg(end)<opts.tol, break; end
    
    
end