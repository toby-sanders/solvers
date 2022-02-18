function [U, out] = PnP3_inpaint(S,R,b,opts,pp)


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

% if numel(S)~=numel(b), error('b and S should be same size'); end


tic;
opts = check_PnP_opts(opts);  % get and check opts

% if isempty(R), R = getBM3D(5/255); end
scl = max(b(:))*2;
if isempty(R), R = getTNRD(scl); end
% mark important constants
tol = opts.tol; 


% initialize everything else
if isfield(opts,'init')
    U = opts.init;
else
    [~,hhat] = makeGausPSF([size(b,1),size(b,2)],3);
    U = real(ifft2(fft2(b).*hhat));
end

V = R(U); % splitting variable, V
mu = opts.mu;beta = opts.beta;
sigma = zeros(size(b,1),size(b,2)); % Lagrange multiplier
out.rel_chg = [];

denomObj = mu*S + beta;
for ii = 1:opts.iter % iterations refers to outer loop
    if nargin == 5
        waitbar(1/3 + (2/3)*ii/opts.iter,pp,'deblurring frame');
    end
    for iInner = 1:5
    Up = U;
    U = (mu*b + beta*V - sigma)./denomObj;
    out.rel_chg = [out.rel_chg; norm(U(:)-Up(:))/norm(Up(:))];
    
    % projected gradient method for inequality constraints
    if opts.nonneg, U = max(real(U),0);
    elseif opts.isreal, U = real(U); end 
    if out.rel_chg(end)<tol, break; end 
    
    % denoising step and multiplier update
    V = R(U + sigma/beta);
    end
    sigma = sigma + beta*(U-V);
    % display for prototyping
    if opts.disp
        fprintf('multiplier update number %i, relChgOut = %g\n',ii,out.rel_chg(end));
        figure(345);
        subplot(2,2,1);imagesc(U);title('U, main variable');colorbar;
        subplot(2,2,2);imagesc(V);title('V, splitting variable');colorbar;
        subplot(2,2,3);imagesc(sigma);title('sigma, multiplier');colorbar;
        subplot(2,2,4);semilogy(out.rel_chg);title('rel change out');
        colormap(gray);
    end    
    
end
out.elapsed_time = toc;
out.rel_UV = myrel(U,V);
if nargin == 5, delete(pp); end


