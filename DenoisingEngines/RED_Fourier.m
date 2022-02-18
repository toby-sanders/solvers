function [U, out] = RED_Fourier(S,R,b,n,opts)


% Written by Toby Sanders @Lickenbrock Tech.
% 2/28/2020

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RED image deconvolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [U, out] = RED_deblur(K,R,b,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||K*u - b||_2^2 + u^T*(u - R(u)) }
%
% where R is an input denoising engine
% the fixed point iteration is implemented

% R can be input as empty, in which case the BM3D denoiser is used

% Inputs: 
%   K: blurring kernel
%   R: an image denoising operator
%   b: data values in vector form
%   opts: structure containing input parameters, 
%       see function check_HOTV_opts.m for these
% Outputs:
%   U: reconstructed signal
%   out: output numerics

tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
opts = check_PnP_opts(opts);  % get and check opts

if isempty(R), R = getBM3D(5/255); end
% mark important constants
tol = opts.tol; 

% initialize everything else
bstr = zeros(p,q,r);
bstr(S) = b;
P = zeros(p,q,r);
P(S) = 1;
Atb = ifftn(bstr)*sqrt(p*q*r);
if numel(opts.init)==1
    U = Atb;
else
    U = opts.init;
end

% V = U; % splitting variable, V
mu = opts.mu; % beta = opts.beta;
lambda = 1/mu;
% beta = 1;
out.rel_chg = [];

for ii = 1:opts.iter % iterations refers to outer loop
    Up = U;
   %  z = fft2(Atb + lambda*V);
   %  U = ifft2(z./(Khat2 + lambda));
   
    % fixed point iteration
    U = ifft2((bstr + lambda*fft2(R(U))/sqrt(p*q*r))./(P + lambda))*sqrt(p*q*r);
    out.rel_chg = [out.rel_chg; norm(U(:)-Up(:))/norm(Up(:))];
    
    % projected gradient method for inequality constraints
    if opts.nonneg, U = max(real(U),0);
    elseif opts.isreal, U = real(U); end 
    if out.rel_chg(end)<tol, break; end 

    % display for prototyping
    if opts.disp
    fprintf('multiplier update number %i, relChgOut = %g\n',ii,out.rel_chg(end));
    figure(345);
    subplot(2,2,1);imagesc(U);title('U, main variable');colorbar;
    % subplot(2,2,2);imagesc(V);title('V, splitting variable');colorbar;
    % subplot(2,2,3);imagesc(sigma);title('sigma, multiplier');colorbar;
    subplot(2,2,4);semilogy(out.rel_chg);title('rel change out');
    colormap(gray);
    end    
    
end
out.elapsed_time = toc;


