function [U,out] = Tikhonov_CG(A,b,n,opts)

% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_u    mu*||Au-b||^2 + ||Du||^2
% using a CG decent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This version of the code is faster than "Tikhonov.m," BUT it does not
% implement the inequality constraints (nonnegitivity)

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% tol - convergence tolerance for CG
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 01/29/2018

% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);

% unify implementation of A, check scaling A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
if opts.scale_A, [A,b] = ScaleA(p*q*r,A,b); end
[D,Dt] = get_D_Dt(opts.order,p,q,r,opts);
B = @(x)A(A(x,1),2) + Dt(D(x))/opts.mu;
c = A(b,2);
tic;
[U,out] = my_local_cgs(B,c,opts.iter,opts.tol);
out.total_time = toc;
U = reshape(U,p,q,r);
out.rel_error = norm(A(U(:),1)-b)/norm(b);
out.Du = norm(D(U(:)))^2;

function [x,out] = my_local_cgs(A,b,iter,tol)
% CG algorithm from wiki

x = zeros(size(b));
out = [];
r = b-A(x);
p = r;
rel_chg = zeros(iter,1);
for i = 1:iter
    Ap = A(p);
    alpha = r'*r./(p'*Ap);
    xp = x;
    x = x + alpha*p;
    rp = r;
    r = r-alpha.*Ap;
    rel_chg(i) = alpha^2*(p'*p)/(x'*x);
    if rel_chg(i) < tol^2; break;end
    beta = r'*r./(rp'*rp);
    p = r+beta.*p;
end
out.iters = i;
out.rel_chg = sqrt(rel_chg(1:i));
