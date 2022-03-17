function [U,uup] = updateU_HOTV(Shhat2,Atb,Dt,U,W,gL,V,muDbeta,beta,opts)

% written by Toby Sanders @Lickenbrock tech
% Last update: 12/2019

% single step updates on U for HOTV L1 ADMM optimization
% this is really just a quadratic minimization step
% The L1 norm is only involved in the shrinkage formula update on W
% For the general case, a gradient descent is used
% For deconvolution and Fourier data the exact minimizer is evaluated

alpha = .01;
[p,q] = size(U);
Up = U; % save previous U for convergence checking and BB steps

% updates for deconvolution 
z = fft2(muDbeta*Atb+reshape(Dt(W)+gL/beta,p,q,1));
U = ifft2(z./(muDbeta*Shhat2 + V));
figure(812);
subplot(2,2,1);imagesc(real(U));
subplot(2,2,2);imagesc(Up);
if sum(U(:))~=0
    U = alpha*U + (1-alpha)*Up;
end
subplot(2,2,3);imagesc(real(U));
% projected gradient method for inequality constraints
if opts.nonneg, U = max(real(U),0);
elseif opts.isreal, U = real(U); end
if opts.max_c, U = min(U,opts.max_v); end
uup = U(:)-Up(:);
