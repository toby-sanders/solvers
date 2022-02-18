function [x,out] = my_cgs(A,b,iter,tol,x0)

% Written by Toby Sanders @ ASU
% School of Math & Stat Sciences
% 12/26/2017

% Decided to try out writing a conjugate gradient solver
% it seems to work well

% Went to wikipedia on CG and the algorithm there is even more efficient...

% check sufficient inputs
if nargin < 2, error('not enough input arguments');
elseif nargin < 3, iter = 50; tol = 1e-5;
elseif nargin < 4, tol = 1e-5; 
elseif nargin < 5, x0 = 0;
end

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end

% convert Ax=b problem to A^T(Ax=b) so new A is SPD
% b = A(b,2);
% A = @(x)A(A(x,1),2);
if x0==0, x = zeros(size(b));
else, x = x0; 
end

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
out.iter = i;