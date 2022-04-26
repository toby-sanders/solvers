function [x,out] = Tikhonov_Nesta(hhat,b,opts)

% a iterative deconvolution algorithm with least squares data fit term and
% Tikhonov regularizer
% normally, this solution can be computed with a simple Wiener filter
% however, to enforce a nonnegativity constraint, you need to iterate
% hence, this function exists and always implements nonnegativity


% the iterations use standard gradient descent with acceleration using the
% Nesterov method/heavy ball approach

% Written by Toby Sanders @Magnetic Insight
% School of Math & Stat Sciences
% 4-26-2022

% set image dimensions
[p,q,r] = size(hhat);
if size(b,1)~=p || size(b,2)~=q || size(b,3)~=r
    error('PSF and image dimensions do not match');
end
mu = opts.mu;
if numel(mu)==1 % expand mu for each frame
    mu = repmat(mu,r,1);
end
opts = check_tik_opts(opts);
bhat = fft2(b);

% create Atb and sum of hhats squared, both with mu included
hhat2 = zeros(p,q);
Atb = hhat2;
for i = 1:r
    hhat2 = hhat2 + mu(i)*abs(hhat(:,:,i)).^2;
    Atb = Atb + mu(i)*ifft2(conj(hhat(:,:,i)).*bhat(:,:,i));
end

% initialize out and x
out.rel_chg = zeros(opts.iter,1);
if ~isempty(opts.init)
    x = opts.init;
else
    x = ifft2(sum(bhat.*hhat,3)./(hhat2+1)); % simple Wiener filter init
end

% get the Fourier regularization matrix
if isfield(opts,'regV')
    if ~isempty(opts.regV)
        V = abs(opts.regV).^2;
    else
        V = my_Fourier_filters(opts.order,opts.levels,p,q,1);
    end
else
    V = my_Fourier_filters(opts.order,opts.levels,p,q,1);
end
% get the Lipchitz constant and corresponding gradient step length
L = max(hhat2(:) + V(:));
tau = (1/L);
filt = hhat2 + V; % filter for the gradient in each iteration

% iterate
xp = x;
tic;
for i = 1:opts.iter
    
    y = x + (i-1)/(i+2)*(x-xp); % new accerated vector
    xp = x; % save old solution for next acceleration
    g = ifft2(fft2(y).*filt) - Atb; % gradient
    x = y - tau*g; % gradient descent from accelerated vector, y        
    x = max(real(x),0);
    
    % check for convergence
    out.rel_chg(i) = sqrt(real(sum((x(:)-xp(:)).^2)/sum(x(:).^2)));
    if out.rel_chg(i) < opts.tol
        out.rel_chg = out.rel_chg(1:i);
        break;
    end
    
end
out.total_time = toc;
out.iters = i;
out.g = g;
