function [U,out] = UPRE(A,b,n,opts,sigma)


% function for finding optimal parameter from UPRE
% i.e. We find the optimal mu by the predictive estimate to the problem
%  min_u  mu||Au-b||^2 + ||Du||^2
% fixed point version

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% August 2018
% Updated: 01/10/2019

Fsig = false;
if nargin<5, Fsig = true; sigma = 1;
elseif sigma==0, Fsig = true; sigma = 1; end
if isfield(opts, 'maxiter'); maxiter = opts.maxiter;
else, maxiter = 50; end
if isfield(opts, 'ntrace'); nQ = opts.ntrace;
else, nQ = 6; end
if isfield(opts,'mu'); mu = opts.mu;
else, mu = 1e-3; end
    
% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
if ~isfield(opts,'tol'), opts.tol = 1e-3; end
opts = check_tik_opts(opts);

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
m = numel(b);
Atb = A(b,2); u = Atb;
[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,Atb);
H = @(mu)@(X)A(A(X,1),2) + Dt(D(X))/mu;
AtQ = randn(m,nQ);
AtQ = A(AtQ,2); % precompute to approx. tr(A H^-1 T^T T H^-1 A^T)
trappV = zeros(p*q*r,nQ);
trappV2 = D(zeros(p*q,r,1));
trappV2 = zeros(numel(trappV2),nQ);

out.mus = mu; out.sigmas = [];
out.errors = []; out.UPREs = []; % store predictive estimator
% if Utrue supplied can compute exact predictive error
if isfield(opts,'Utrue'), out.eetrue = []; end
for i = 1:maxiter
    % solve Tikhonov reg. problem for each mu
    uo = u;
    B = H(mu);
    [u,~] = my_local_cgs(B,Atb,opts.iter,1e-7);
    re = A(u,1) - b;
    errors = myrel(u,uo);
    out.errors = [out.errors;errors];
    if errors <= opts.tol, break; end
    % compute D H^-1 A^T Q
    % then approx. trace of A H^-1 D^T D H^-1 A^T
    % and trace of A H^-1 A^T
    for j = 1:nQ
         trappV(:,j) = my_local_cgs(B,AtQ(:,j),opts.iter,1e-7);
         trappV2(:,j) = col(D(trappV(:,j))); % trace vector (D H^-1 A^T Q)
    end    
    trapp = sum(col(trappV.*AtQ))/nQ; % tr(A H^-1 A^T)
    trapp2 = sum(col(trappV2.*trappV2))/nQ;
    
    % update sigma if unknown and only if m>trapp
    if Fsig && m>trapp, sigma = sqrt(re'*re/(m-trapp)); end
    % compute UPRE
    out.UPREs = [out.UPREs;1/m*(re'*re)+2*sigma^2/m*trapp-sigma^2];
    if isfield(opts,'Utrue')
        out.eetrue = [out.eetrue;1/m*norm(A(u-opts.Utrue,1))^2];
    end    
    
    % new mu, updated based on the fixed point
    % (u'T'TH^{-1}T'Tu)/(sigma^2*trace(AH^{-1}T'TH^{-1}A'))
    tmp = Dt(D(u));
    tmp2 = my_local_cgs(B,tmp,opts.iter,1e-7);
    mu = real((tmp'*tmp2)/(sigma^2*trapp2));
    out.mus = [out.mus;mu];out.sigmas = [out.sigmas;sigma];
end
U = reshape(u,p,q,r);

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

