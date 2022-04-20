function [x,out] = Tikhonov_Nesta(A,b,n,opts)

% this version uses standard gradient descent with acceleration using the
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
mu = opts.mu;

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end
%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;


% initialize out and x
out.rel_chg = zeros(opts.iter,1);
x = zeros(p*q*r,1);

% get the step length for the objective function
[tau,out.tauStuff] = getStepLength(A,n);
L = my_Fourier_filters(opts.order,opts.levels,p,q,r);
L = mu/tau + max(L(:)); % lipchitz constant for combined operators
tau = 1/L;

[D,Dt] = get_D_Dt(opts.order,p,q,r,opts);
[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg, error('D and Dt do not appear consistent'); end
clear flg;

xp = x;
tic;
for i = 1:opts.iter
    
    y = x + (i-1)/(i+2)*(x-xp); % new accerated vector
    xp = x;
    g = mu*(A(A(y,1)-b,2)) + Dt(D(y));
    x = y - tau*g; % gradient descent from accelerated vector, y    
    
    if opts.nonneg, x = max(real(x),0);
    elseif opts.isreal, x = real(x); end   
    
    % check for convergence
    out.rel_chg(i) = norm(x-xp)/norm(xp);
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