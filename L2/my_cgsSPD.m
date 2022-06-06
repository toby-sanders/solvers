function [x,out] = my_cgs(A,b,iter,tol)

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
end

% % unify implementation of A
if ~isa(A,'function_handle'), A = @(u) f_handleA(A,u,1); end

% this version passes a matrix operator A which is already SPD



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