function [U, out] = PnP3_deblur(K,R,b,opts)


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
opts = check_PnP_opts(opts);  % get and check opts

% if isempty(R), R = getBM3D(5/255); end
scl = max(b(:))*2;
if isempty(R), R = getLTBM3D; end
% mark important constants
tol = opts.tol; 
Khat = fft2(K);
Khat2 = sum(Khat.*conj(Khat),3);
nF = size(b,3);
if size(Khat,3)~=nF, Khat2 = Khat2*nF; end

% initialize everything else
Atb = real(sum(ifft2(fft2(b).*conj(Khat)),3));
if isfield(opts,'init')
    U = opts.init;
else
    U = Atb/nF;
end

V = U; % splitting variable, V
mu = opts.mu;beta = opts.beta;
sigma = zeros(size(b,1),size(b,2)); % Lagrange multiplier
out.rel_chg_inn = [];out.rel_chg_out = [];
ii=0;
while numel(out.rel_chg_inn)<opts.iter % iterations refers to outer loop
    ii = ii+1;
    for jj = 1:opts.inner_iter
        Up = U;
        z = fft2(mu*Atb + beta*V - sigma);
        U = ifft2(z./(mu*Khat2 + beta));


        % projected gradient method for inequality constraints
        if opts.nonneg, U = max(real(U),0);
        elseif opts.isreal, U = real(U); end 
        % if out.rel_chg(end)<tol, break; end 

        % denoising step and multiplier update
        V = R(U + sigma/beta,opts.sigma0/sqrt(beta));
    
        out.rel_chg_inn = [out.rel_chg_inn; norm(U(:)-Up(:))/norm(Up(:))];
        if (out.rel_chg_inn(end) < opts.tol) || ii>= opts.iter, break; end 
    end
      
    sigma = sigma + beta*(U-V);
    
    gpuID = gpuDevice(2);
    reset(gpuID);
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
out.V = V;
out.sigma = sigma;
out.beta = beta;
out.mu = mu;


