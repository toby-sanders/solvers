function [U,params] = updateU_HOTV(A,b,Dt,U,Uc,W,gL,mu,beta,params,opts)

% written by Toby Sanders @Lickenbrock tech
% Last update: 12/2019

% single step updates on U for HOTV L1 ADMM optimization
% this is really just a quadratic minimization step
% The L1 norm is only involved in the shrinkage formula update on W
% For the general case, a gradient descent is used
% For deconvolution and Fourier data the exact minimizer is evaluated

Up = U; % save previous U for convergence checking and BB steps
switch params.mode % handling various update types
    case 'deconv' % updates for deconvolution 
        % bb = fftn(mu*params.Atb+reshape(Dt(beta*W)+gL,p,q,r));
        % U = ifftn(bb./(mu*A + beta*params.V));
        ratio = b./ifftn(fftn(U).*A);
        numerObj = ifftn(fftn(ratio).*conj(A));
        denomObj = reshape(mu*sum(col(ifftn(A))) + beta*Dt(Uc-W) - gL,size(A,1),size(A,2));
        U = U.*numerObj./denomObj;       
end
% projected gradient method for inequality constraints
if opts.nonneg, U = max(real(U),1e-3);
elseif opts.isreal, U = real(U); end
if opts.max_c, U = min(U,opts.max_v); end
params.uup = U(:)-Up(:);
