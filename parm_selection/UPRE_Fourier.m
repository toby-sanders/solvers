function [U,out] = UPRE_Fourier(S,b,n,opts,sigma)


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
if isfield(opts,'mu'); mu = opts.mu;
else, mu = 1e-3; end
    
% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
opts = check_tik_opts(opts);nlev = opts.levels;
m = numel(b); k = opts.order;
[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,b);

% store eigenvalues of regularization operator 
vx = zeros(nlev,q); vy = zeros(nlev,p); vz = zeros(nlev,r);
for i = 1:nlev
    vx(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,q-1,q)/q).^(2*k+2))./...
            (sin(pi*linspace(0,q-1,q)/q).^2)*(2^(2-k-i)/nlev)^2;
    vy(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,p-1,p)/p).^(2*k+2))./...
        (sin(pi*linspace(0,p-1,p)/p).^2)*(2^(2-k-i)/nlev)^2;
    vz(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,r-1,r)/r).^(2*k+2))./...
        (sin(pi*linspace(0,r-1,r)/r).^2)*(2^(2-k-i)/nlev)^2;
end
vx(:,1) = 0; vy(:,1) = 0; vz(:,1) = 0;
vx = sum(vx,1); vy = sum(vy,1); vz = sum(vz,1); 
V = vy' + vx + reshape(vz,1,1,r);
V2 = zeros(p,q,r); V2(S) = 1;
% stretch b
bstr = zeros(prod(n),1);
bstr(S) = b; bstr = reshape(bstr,p,q,r);
u = ones(prod(n),1);

out.mus = mu; out.sigmas = [];
out.errors = []; out.UPREs = []; % store predictive estimator
% if Utrue supplied can compute exact predictive error
if isfield(opts,'Utrue'), out.eetrue = zeros(maxiter,1); end
for i = 1:maxiter
    % solve Tikhonov reg. problem for each mu
    uo = u; Fu = bstr./(1+ V/mu);
    u = sqrt(prod(n))*col(real(ifftn(Fu)));
    re = col(bstr-Fu); errors = myrel(u,uo);
    out.errors = [out.errors;errors];
    if errors <= 1e-3, break; end
    % exact traces of H^-1 D^T D H^-1 and   H^-1
    trHiAA = sum(1./(col(V(S))/mu + 1));
    trBIG = sum(V(S)./(1+V(S)/mu).^2);
    % trBIG = sum(col(V)./(1+col(V)/mu).^2);
    
    % update sigma if unknown and only if m>trapp
    if Fsig && m>trHiAA, sigma = sqrt(re'*re/(m-trHiAA)); end
    % compute UPRE
    out.UPREs = [out.UPREs;1/m*(re'*re)+2*sigma^2/m*trHiAA-sigma^2];
    if isfield(opts,'Utrue')
        out.eetrue(i) = 1/m*norm(A(u-opts.Utrue,1))^2;
    end    
    
    % new mu, updated based on the fixed point
    % (u'T'TH^{-1}T'Tu)/(sigma^2*trace(AH^{-1}T'TH^{-1}A'))
    tmp = V.*Fu; tmp2 = tmp./(1+V/mu);
    mu = sum(col(real(conj(tmp).*tmp2)))/(sigma^2*trBIG);
    out.mus = [out.mus;mu];out.sigmas = [out.sigmas;sigma];
end
U = reshape(u,p,q,r);

