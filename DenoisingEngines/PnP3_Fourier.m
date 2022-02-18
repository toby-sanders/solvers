function [U, out] = PnP3_Fourier(S,R,b,n,opts)


% Written by Toby Sanders @Lickenbrock Tech.
% 10/21/2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plug and Play image deconvolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [U, out] = PnP3_deblur(K,R,b,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||K*u - b||_2^2 + R(u) }
%
% where R is an input denoising operator, e.g. if V is a noisy image, the
% R(V) is the denoised image
% Generally speaking, this formulation does not makes sense, since R is not
% a functional, however, the method derived from the ADMM approach in an
% approach known as "plug-and-play" (PnP)

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

% if isempty(R), R = getBM3D(5/255); end
scl = max(b(:))*2;
if isempty(R), R = getTNRD(scl); end
% mark important constants
tol = opts.tol; 
P = zeros(p,q,r);
P(S) = 1;
bstr = zeros(p,q,r);
bstr(S) = b;

% initialize everything else
Atb = ifft2(bstr);% *sqrt(p*q*r);
if ~isfield(opts,'init')
    U = Atb;
else
    U = opts.init;
end

V = U; % splitting variable, V
mu = opts.mu;beta = opts.beta;
sigma = zeros(p,q,r); % Lagrange multiplier
out.rel_chg = [];




for ii = 1:opts.iter % iterations refers to outer loop

    Up = U;
    % z = fft2(mu*Atb + beta*V - sigma);
    % U = ifft2(z./(mu*Khat2 + beta));
    z = fft2(mu*Atb + beta*V - sigma);
    U = ifft2(z./(mu*P + beta));
    out.rel_chg = [out.rel_chg; norm(U(:)-Up(:))/norm(Up(:))];
    
    % projected gradient method for inequality constraints
    if opts.nonneg, U = max(real(U),0);
    elseif opts.isreal, U = real(U); end 
    if out.rel_chg(end)<tol, break; end 

    % denoising step and multiplier update
    V = R(real(U + sigma/beta),opts.sigma0/sqrt(beta));

    sigma = sigma + beta*(U-V);
    % display for prototyping
    if opts.disp
        fprintf('multiplier update number %i, relChgOut = %g\n',ii,out.rel_chg(end));
        figure(345);
        % subplot(2,2,1);imagesc(U);title('U, main variable');colorbar;
        subplot(2,2,2);imagesc(V);title('V, splitting variable');colorbar;
        % subplot(2,2,3);imagesc(sigma);title('sigma, multiplier');colorbar;
        subplot(2,2,4);semilogy(out.rel_chg);title('rel change out');
        colormap(gray);
    end    
    
end
out.elapsed_time = toc;
out.rel_UV = myrel(U,V);



