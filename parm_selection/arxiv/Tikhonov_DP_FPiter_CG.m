function [x,out] = Tikhonov_DP_FPiter(A,b,n,sigma,opts)

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
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = p*q*r;
opts = check_tik_opts(opts);

% mark important constants
mu = opts.mu;
iter = opts.iter;
tol = opts.tol;
k = opts.order;

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end

%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

% check scaling A
% if opts.scale_A, [A,b] = ScaleA(p*q*r,A,b); end

% initialize out and x
out.rel_error = zeros(iter,1);
out.nrm_reg = zeros(iter,1);
out.obj_func = zeros(iter,1);
out.rel_chg = zeros(iter,1);
out.mu = mu; out.neg_mu_cnt = 0;
x = zeros(p*q*r,1);


[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,A(b,2));
bb = A(b,2);

nrmb = norm(b);
m = numel(b);
fprintf('Running Tikhonov solver...\n');
for i = 1:20
    B = @(x)A(A(x,1),2) + Dt(D(x))/mu;
    [x,~] = my_local_cgs(B,bb,500,1e-7);
    % figure(73);plot(x);title(i);pause;
    dd = D(x);   % Dx
    Ax = A(x,1);
    
    % update mu
    mup = mu;
    mu = dd(:)'*dd(:)/(nrmb^2-b'*Ax-m*sigma^2);
    if mu<0
        mu = mup/10;
        out.neg_mu_cnt = out.neg_mu_cnt +1;
    end
    out.mu = [out.mu;mu];
end
fprintf('total iterations = %i\n\n',i);
% output final solution
x = reshape(x,p,q,r);




function [x,out] = my_local_cgs(A,b,iter,tol)
% CG algorithm from wiki

x = zeros(size(b));
out = [];
r = b-A(x);
p = r;
err = zeros(iter,1);
for i = 1:iter
    Ap = A(p);
    alpha = r'*r./(p'*Ap);
    x = x + alpha*p;
    rp = r;
    r = r-alpha.*Ap;
    err(i) = alpha^2*(p'*p)/(x'*x);
    if err(i) < tol^2; break;end
    beta = r'*r./(rp'*rp);
    p = r+beta.*p;
end
out.iters = i;
out.err = err(1:i);