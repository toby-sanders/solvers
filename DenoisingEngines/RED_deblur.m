function [U, out] = RED_deblur(K,R,b,opts)


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
opts = check_PnP_opts(opts);  % get and check opts

scl = max(b(:))*2;
if isempty(R), R = getTNRD(scl); end

% mark important constants
tol = opts.tol; 
Khat = fft2(K);
Khat2 = sum(Khat.*conj(Khat),3);
nF = size(b,3);
if size(Khat,3)~=nF, Khat2 = Khat2*nF; end

% initialize everything else
Atb = real(sum(ifft2(fft2(b).*conj(Khat)),3));
U = Atb/nF;
Up = U;

% V = U; % splitting variable, V
mu = opts.mu; % beta = opts.beta;
lambda = 1/mu;
% beta = 1;
out.rel_chg = [];
out.objF = [];
for ii = 1:opts.iter
    if opts.fast, NestAlpha = (ii-1)/(ii+2);
    else, NestAlpha = 0;
    end
    
    % evaluate intermediate image and save previous for Nest. acceleration
    V = U + NestAlpha*(U-Up);
    Up = U;
    
    % iterate
    Rv = R(V); % denoise
    U = ifft2(fft2(Atb + lambda*Rv)./(Khat2 + lambda)); % fixed point iteration
    out.rel_chg = [out.rel_chg; norm(U(:)-Up(:))/norm(Up(:))]; 
    
    % use this to check the value of the objective function
    objF = norm(col(ifft2(fft2(V).*Khat) - b),2)^2 +...
        sum(V(:).*(V(:) - Rv(:)));
    out.objF = [out.objF; objF];
    
    % projected gradient method for inequality constraints
    if opts.nonneg, U = max(real(U),0);
    elseif opts.isreal, U = real(U); end 
    if out.rel_chg(end)<tol, break; end 

    % display for prototyping
    if opts.disp
        figure(345);colormap(gray);
        subplot(1,2,1);imagesc(U);title('U, main variable');colorbar;
        subplot(1,2,2);semilogy(out.rel_chg);title('rel change out');
    end    
    
end
out.elapsed_time = toc;


