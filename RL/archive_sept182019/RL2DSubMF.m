function [x,out] = RL2DSubMF(h,g,b,M,opts)

% g is the pixel averaging kernel (low resolution)
% h is the main convolution kernel
% b is the subselected data in vector form
% with multiple frames, b holds multiple columns of data
% M is the image mask pointing to the pixel locations
% opts - stucture with options, the fields being:
%   iter - maximum iteration
%   tol - convergence tolerance
%   x - initial guess

% set up options
if nargin<5
    opts.iter = 100;
end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if isfield(opts,'x'), x = opts.x; end
if ~isfield(opts,'disp'), opts.disp = 0; end
epsilon = 1e-4;


[p,q] = size(M); % image reconstruction size
k = size(b,2); % frame count
% stretch data into full image size
bstr = zeros(p*q,k); 
S = find(M);  
for i = 1:k, bstr(S,i) = b(:,i); end
bstr = reshape(bstr,p,q,k);

% precompute convolution filters
hhat = fft2(h,p,q);
ghat = fft2(g,p,q);
Ghat = ghat.*hhat;

% initialize x
if ~isfield(opts,'x')
   %  x = ifft2(ghat.*fft2(bstr(:,:,1)));
    % x = ones(p,q)*100;
    
   %  x = sum(ifft2(Ghat.*fft2(bstr)),3);
    % x = ifft2(Ghat(:,:,1).*fft2(bstr(:,:,1)));
    % x = sum(bstr,3) + epsilon;
    x = ones(p,q)*100; %rand(p,q)*100;
end

% precompute denomenator G*x for all frames and the normalization factor 
Fx = fft2(x);
Gx = zeros(p,q,k);
normalizer = zeros(p,q);% ifft2(fft2(double(M)).*conj(Ghat(:,:,1)));
for i = 1:k
    normalizer = normalizer + ifft2(fft2(double(M)).*conj(Ghat(:,:,i)));
    Gx(:,:,i) = ifft2(Fx.*Ghat(:,:,i));
end

% iterate
out.rel_error = [];
for i = 1:opts.iter
    % compute main ratio and correction factor
    xp = x;     
    ratio = bstr./(Gx+epsilon);
    cfactor = sum(ifft2(fft2(ratio).*conj(Ghat)),3)./(normalizer+epsilon);  
    
    % update x and then Gx
    x = x.*cfactor; 
    Fx = fft2(x);
    for j = 1:k
        Gx(:,:,j) = ifft2(Fx.*Ghat(:,:,j));
    end
    out.rel_error = [out.rel_error;myrel(x,xp)];   
    if opts.disp
        figure(75);
        subplot(2,2,1);imagesc(real(x));title('x');colorbar;
        subplot(2,2,2);imagesc(real(cfactor));title('correction');colorbar;
        subplot(2,2,3);imagesc(Gx);title('G*x');colorbar;
        subplot(2,2,4);imagesc(bstr(:,:,1));title('b');colorbar;
    end
end