function [U,out] = UPRE_guess(A,b,n,opts,sigma,mumin,mumax,Nmu)


% function for finding optimal parameter from UPRE
% i.e. We find the optimal mu by the predictive estimate to the problem
%  min_u  mu||Au-b||^2 + ||Du||^2
% this version runs the "global" brute force search

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 05/10/2018

if nargin<5, error('must specify sigma');
elseif nargin<6, mumin = 1e-5;mumax = 1e5;Nmu = 20;
elseif nargin<7, mumax = 1e5; Nmu = 20;
elseif nargin<8, Nmu = 20;
end
    
% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
nQ = 6; % number of probing vectors for randomized trace approx.
opts = check_tik_opts(opts);

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
m = numel(b);
Atb = A(b,2);
[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,Atb);
H = @(mu)@(X)A(A(X,1),2) + Dt(D(X))/mu;
Q = randn(p*q*r,nQ); % probing vectors
AAQ = A(A(Q,1),2); % precompute to approx. tr(H^-1 A'*A)


mus = linspace(log10(mumin),log10(mumax),Nmu);
mus = 10.^mus;
recs = zeros(p*q*r,Nmu);
out.UPREs = zeros(Nmu,1); % store predictive estimator
% if Utrue supplied can compute exact predictive error
if isfield(opts,'Utrue'), out.eetrue = zeros(Nmu,1); end
for i = 1:Nmu
    % solve Tikhonov reg. problem for each mu
    opts.mu = mus(i);
    B = H(opts.mu);
    [recs(:,i),~] = my_local_cgs(B,Atb,opts.iter,opts.tol);
    re = A(recs(:,i),1) - b;
    
    %trace approximation: compute H^-1 Q
    trapp = zeros(p*q*r,nQ);
    for j = 1:nQ
        trapp(:,j) = my_local_cgs(B,Q(:,j),opts.iter,opts.tol);
    end    
    trapp = sum(col(trapp.*AAQ))/nQ;

    % compute UPRE
    out.UPREs(i) = 1/m*(re'*re)+2*sigma^2/m*trapp-sigma^2;
    if isfield(opts,'Utrue')
        out.eetrue(i) = 1/m*norm(A(recs(:,i)-opts.Utrue,1))^2;
    end
end
out.mus = mus;
[~,muopt] = min(out.UPREs);
out.muoptimal = mus(muopt);
out.recs = recs;
U = reshape(recs(:,muopt),p,q,r);

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
out.iter = i;
out.err = err(1:i);

