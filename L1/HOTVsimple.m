function [U,out] = HOTVsimple(A,b,n,opts)

% testing out a simple gradient decent for HOTV using the Euler-lagrange
% equations for the isotropic hotv regularization.
% for some reason, though it does a fine job of regularizing, 
% it doesn't appear to capture any sparsity...

tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);

opts = check_HOTV_opts(opts);  % get and check opts

% mark important constants
tol = opts.tol; 
k = opts.order; n = p*q*r;
wrap_shrink = opts.wrap_shrink;
epsilon = 1e-3;
% check that A* is true adjoint of A
% check scaling of parameters, maximum constraint value, etc.
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end; clear flg;
if opts.scale_A, [A,b] = ScaleA(n,A,b); end
[~,scl] = Scaleb(b);
if opts.scale_mu, opts.mu = opts.mu*scl; end
opts.beta = opts.beta*scl;
if round(k)~=k || opts.levels>1, wrap_shrink = true; end

% initialize everything else
Atb = A(b,2); % A'*b
[U,mu,beta,~,~,muDbeta,sigma,delta,gL,ind,out] ...
    = get_HOTV(p,q,r,Atb,A(Atb,1),scl,opts,k,b,wrap_shrink);

U = U(:);
Aub = A(U,1)-b;
out.objf_val = mu*Aub'*Aub + computeTVnorm(reshape(U,p,q,r),k);
out.rel_err = [];out.rel_chg  = [];
ii = 0;  % main loop
while ii <= opts.iter
    ii = ii + 1;
    g = mu*A(Aub,2) + col(functionalDerHOTV(reshape(U,p,q),epsilon,k));
    tau = .01;
    if ii>1
       uup = U-up;
       tau = uup'*uup/(uup'*(g-gp));
    end
    up = U;
    gp = g;
    U = U - tau*g;
    if opts.nonneg, U = max(U,0); end
    Aub = A(U,1)-b;
    Aub2 = Aub'*Aub;
    out.rel_err = [out.rel_err;Aub2];
    objF = mu*Aub2 + computeTVnorm(reshape(U,p,q,r),k);
    out.objf_val = [out.objf_val;objF];
    out.rel_chg = [out.rel_chg;myrel(U,up)];
    if out.rel_chg(end)<tol, break;end
%     figure(85);
%     subplot(2,2,1);plot(U);
%     subplot(2,2,2);semilogy(out.rel_chg);
%     
end
out.rel_err = sqrt(out.rel_err)/norm(b);
U = reshape(U,p,q,r);
out.elapsed_time = toc;
    
    
    


function uTV = computeTVnorm(u,order)
if order == 0, uTV = sum(sum(abs(u))); return; end

ux = diff(u,order,2);
uy = diff(u,order,1);
uTV = sum(sum(sqrt(ux(1:end-order,:).^2 + uy(:,1:end-order).^2)));