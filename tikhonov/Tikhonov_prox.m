function [x,out] = Tikhonov_prox(A,b,n,opts)

% this version uses a proximal gradient method with acceleration via the
% Nesterov method/heavy ball approach


% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest descent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% tol - convergence tolerance
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/1/2018

% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = p*q*r;

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end

%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

% get the step length, which is 1/lambda_max(A^T*A)
[tau,out.tauStuff] = getStepLength(A,n);
tau = tau*.95;

% initialize out and x
out.rel_chg = zeros(opts.iter,1);
x = zeros(n,1);
xp = x;

% construct the denoising filter used in the proximal step
V = my_Fourier_filters(opts.order,opts.levels,p,q,r);
filt = 1./(1 + tau*V/opts.mu);

tic;
for i = 1:opts.iter
      
    
    % create acceleration vector and compute gradient from there
    alpha = (i-1)/(i+2);
    y = x + alpha*(x-xp);
    xp = x; % save previous solution
    g = A(A(y,1)-b,2); % gradient

    % proximal gradient steps
    x = y - tau*g; % gradient
    x = reshape(x,p,q,r);
    x = ifftn(fftn(x).*filt); % proximity operation

    
    % projected gradient method for inequality constraints
    if opts.nonneg, x = max(real(x),0);
    elseif opts.isreal, x = real(x); end
    out.rel_chg(i) = norm(x-xp)/norm(xp);
    
    % check for convergence
    if out.rel_chg(i) < opts.tol
        out.rel_chg = out.rel_chg(1:i);
        break;
    end
  
end
out.total_time = toc;
out.iters = i;
out.g = g;
% output final solution
x = reshape(x,p,q,r);