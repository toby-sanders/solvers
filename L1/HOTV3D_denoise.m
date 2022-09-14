function [U, out] = HOTV3D(I,opts)

% written by Toby Sanders @Lickenbrock tech
% Last update: 12/2019

% HOTV L1 regularization and multiscale variants
% optimized using the ADMM
% this code has been unified to handle deconvolution and Fourier data



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   General Problem Description     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [U, out] = HOTV3D(A,b,n,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||u - I||_2^2 + ||D^k u||_1 }
%
% where D^k is kth order finite difference.
% Multiscale finite differences D^k can also be used.
% To see how to modify these settings read the file "check_HOTV_opts.m"
%
% The problem is modified using variable splitting and this algorithm 
% works with the following augmented Lagrangian function: 
%
%      min_{u,w} {mu/2 ||Au - b||_2^2 + beta/2 ||D^k u - w ||_2^2 
%               + ||w||_1  - (sigma , D^k u - w) }
%
% sigma is a Lagrange multiplier
% Algorithm uses alternating direction minimization over u and w.
%

%
% Inputs: 
%   I: noisy image or volume
%   opts: structure containing input parameters, 
%       see function check_HOTV_opts.m for these
%
% Outputs:
%   U: reconstructed signal
%   out: output numerics

[p,q,r] = size(I);
sclI = max(abs(I(:)));
I = I/sclI;
opts = check_HOTV_opts(opts);  % get and check opts

% mark important variables
tol = opts.tol; 
tol_inn = max(tol,1e-5);  % inner loop tolerance isn't as important
k = opts.order;
[D,Dt] = FD3D(k,p,q,r); % get the finite difference operators

% initialize solution, splitting variables and multipliers
U = zeros(p,q,r); 
sigma = D(U);  W = sigma; Uc = W;
if size(opts.init,1) == p, U = opts.init; end
gL = zeros(p*q*r,1); % initialize gradient on lagrange multiplier

% For each case, initialize variables used in code that are not updated
V = my_Fourier_filters(k,opts.levels,p,q,r);

% initialize everything else
out.rel_chg_inn = []; out.objf_val = [];
beta = opts.beta;
mu = opts.mu;


ii = 0;  % main loop
while numel(out.rel_chg_inn) < opts.iter
    ii = ii + 1; % count number of multiplier updates
    for jj = 1:opts.inner_iter

        % alternating minimization steps between U and W
        Up = U;
        bb = fftn(mu*I+reshape(Dt(beta*W)+gL,p,q,r));
        U = real(ifftn(bb./(mu + beta*V))); % image update
        if opts.nonneg, U = max(U,0); end

        % shrink the transform coefficients (splitting variable update)
        Uc = D(U);
        W = shrinkage(Uc,opts.L1type,beta,sigma);
        
        % check for convergence
        rel_chg_inn = norm(U(:)-Up(:))/norm(U(:));
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        
        % check inner loop convergence
        if (rel_chg_inn < tol_inn) || numel(out.rel_chg_inn)>=opts.iter, break; end 

    end
    % convergence criteria
    if jj<opts.inner_iter && rel_chg_inn<tol && ii>4, break;end
    
    % update multipliers and gradient
    sigma = sigma - beta*(Uc-W);
    gL = Dt(sigma);%  + A(delta,2);    
end
out.mu = mu;
% out.elapsed_time = toc;
out.total_iter = numel(out.rel_chg_inn);
U = U*sclI;

function W = shrinkage(Uc,L1type,beta,sigma)
% update step on W using shrinkage formulas
if strcmp(L1type,'anisotropic')
    W = Uc - sigma/beta;
    W = max(abs(W) - 1/beta, 0).*sign(W);    
elseif strcmp(L1type,'isotropic')
    W = Uc - sigma/beta;
    Ucbar_norm = sqrt(W(:,1).*conj(W(:,1)) + ...
        W(:,2).*conj(W(:,2)) + W(:,3).*conj(W(:,3)));
    Ucbar_norm = max(Ucbar_norm - 1/beta, 0)./(Ucbar_norm+eps);
    W(:,1) = W(:,1).*Ucbar_norm;
    W(:,2) = W(:,2).*Ucbar_norm;
    W(:,3) = W(:,3).*Ucbar_norm;
else
    error('Somethings gone wrong.  L1type is either isotropic or anisotropic');
end


