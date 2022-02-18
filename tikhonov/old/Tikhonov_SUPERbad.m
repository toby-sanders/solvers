function [x,out] = Tikhonov_SUPER(I,K,hhat,opts)

% super resolution function with Tikhonov formulation


% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest decent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% tol - convergence tolerance for CG
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/1/2018

% set image dimensions
[m,n] = size(I);
M = m*K;
N = n*K;

opts = check_tik_opts(opts);

% mark important constants
mu = opts.mu;
iter = opts.iter;
tol = opts.tol;

g = zeros(M,N);
g([1:K],[1:K]) = 1/K^2;
g = fraccircshift(g,[-K/2 + 1/2, -K/2 + 1/2]);
ghat = fft2(g);
ghat = ghat.*hhat;
A = @(input1,input2)myLocalSuperResOpers(K,ghat,input1,input2);
% A = getSuperResDeblurOpers(M,N,K,hhat,ghat);
x = zeros(M,N);

Atb = A(I,2);
V = my_Fourier_filters(opts.order,1,M,N,1);

tau = min(mu/max(V(:)),K);
out.rel_chg = zeros(iter,1);
for i = 1:iter
    gA = A(A(x,1),2);
    gT = ifft2(fft2(x).*V)/mu;
    grad = real(tau*(gA + gT - Atb));
    x = x - grad;
    out.rel_chg(i) = norm(grad(:))/norm(x(:)); 
end



function y = myLocalSuperResOpers(K,fhat,U,mode)
    switch mode
        case 1
            % U = reshape(U,p,q);
            y = (ifft2(fft2(U).*fhat));
            y = y(1:K:end,1:K:end);
        case 2
            % U = reshape(U,p/K,q/K);
            y = zeros(size(U,1)*K,size(U,2)*K);
            y(1:K:end,1:K:end) = U;
            y = (ifft2(fft2(y).*conj(fhat)));
    end

