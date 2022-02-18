function [U, out] = HOTV3D_general(A,b,n,opts)

% the general form of HOTV, where A is a general operator
% intermediate variables are not saved in this version

% Modifications by Toby Sanders @ASU
% School of Math & Stat Sciences
% Last update: 12/2019
%
%
% This code has been modified to solve l1 penalty problems with
% higher order TV operators.  Several small bugs and notation
% changes have been made as well.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [U, out] = HOTV3D(A,b,n,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||Au - b||_2^2 + ||D^k u||_1 }
%
% where D^k is kth order finite difference.
% Multiscale finite differences D^k can also be used.
% To see how to modify these settings read the file "check_HOTV_opts.m"
%
% The problem is modified using variable splitting and this algorithm 
% works with the following augmented Lagrangian function: 
%
%      min_{u,w} {mu/2 ||Au - b||_2^2 + beta/2 ||D^k u - w ||_2^2 
%               + ||w||_1 - (delta , Au - b ) - (sigma , D^k u - w) }
%
% delta and sigma are Lagrange multipliers
% Algorithm uses alternating direction minimization over u and w.
% by default delta = 0.
%
% This algorithm was originally authored by Chengbo Li at Rice University
% as a TV solver called TVAL3.
% original code and description can be found here: 
% http://www.caam.rice.edu/~optimization/L1/TVAL3/
%
% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   b: data values in vector form
%   n: image/ signal dimensions in vector format
%   opts: structure containing input parameters, 
%       see function check_HOTV_opts.m for these
%
%
% Outputs:
%   U: reconstructed signal
%   out: output numerics


tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);

opts = check_HOTV_opts(opts);  % get and check opts

% mark important constants
tol = opts.tol; 
tol_inn = max(tol,1e-5);  % inner loop tolerance isn't as important
k = opts.order; n = p*q*r;
wrap_shrink = opts.wrap_shrink;
L1type = opts.L1type;

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
[U,mu,beta,~,~,~,sigma,delta,gL,ind,out,D,Dt] ...
    = get_HOTV(p,q,r,Atb,A(Atb,1),scl,opts,k,b,wrap_shrink);
nrmb = norm(b);
Uc = D(U);

% first shrinkage step
W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind);

% gA and gD are the gradients w.r.t. u of ||Au-b||^2 and ||Du-w||^2
gD = Dt( Uc - W);
Au = A(U(:),1);
gA = A(Au,2) - Atb;% gA = A'(Au-b)
g = beta*gD + mu*gA - gL; % full obj gradient
Aub = Au-b;
Aub2 = Aub'*Aub;

out.mu = [out.mu; mu];
out.objf_val = mu/2*Aub2+sum(abs(Uc(:)));
objf_best = out.objf_val;out.Ubest = U;
ii = 0;  % main loop
while numel(out.f) <= opts.iter
    ii = ii + 1;
    % optimal step length at the 1st iteration
    gc = D(reshape(g,p,q,r));       
    dDd = sum(col(gc.*conj(gc)));
    Ag = A(g,1);
    tau = abs((g'*g)/(beta*dDd + mu*(Ag')*Ag));
    % check for convergence after the Lagrange multiplier updates
    if ii~=1
        rel_chg_out = norm(tau*g)/norm(U(:));
        out.rel_chg_out = [out.rel_chg_out; rel_chg_out];
        if rel_chg_out < tol, break; end
    end
    
    for jj = 1:opts.inner_iter
        % compute step length, tau
        if jj~=1
            % BB-like step length
            dgA = gA - gAp;   
            dgD = gD - gDp;                    
            ss = uup'*uup;                      
            sy = uup'*(beta*dgD + mu*dgA);       
            tau = abs(ss/max(sy,eps));   
        end

        % keep previous values for backtracking & computing next tau
        Up = U; gAp = gA; gDp = gD;
        U = U(:) - tau*g; % gradient decent
        
        % projected gradient method for inequality constraints
        if opts.nonneg, U = max(real(U),0);
        elseif opts.isreal, U = real(U); end
        if opts.max_c, U = min(U,opts.max_v); end
        U = reshape(U,p,q,r);  Uc = D(U);

        W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind);

        uup = U - Up; uup = uup(:);           % uup: pqr
        rel_chg_inn = norm(uup)/norm(Up(:));
        
        out.objf_val = [out.objf_val; mu/2*Aub2+sum(abs(Uc(:)))];
        out.tau = [out.tau; tau]; 
        out.mu = [out.mu; mu];
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        if out.objf_val(end) < objf_best
            objf_best = out.objf_val(end);
            out.Ubest = U;
            out.Ubest_iter = numel(out.f)-1;
        end
        if opts.store_soln
            out.Uall(:,:,numel(out.f)) = U;
        end
      
        % recompute gradient
        gD = Dt(Uc - W);
        Au = A(U(:),1);
        gA = A(Au,2) - Atb;% gA = A'(Au-b)
        Aub = Au-b;
        Aub2 = Aub'*Aub;
        g = beta*gD + mu*gA - gL;
        % check inner loop convergence
        if (rel_chg_inn < tol_inn) || numel(out.f)>=opts.iter, break; end 

    end
    % my convergence criteria
    if jj<3 && rel_chg_inn < tol && ii>4, break;end
    
    % update multipliers
    sigma = sigma - beta*(Uc-W);
    Aub = Au-b;
    if opts.data_mlp, delta = delta - mu*Aub; end
    if opts.disp, fprintf(' Lagrange mlps update number %i\n',ii);end
    
    % update function value, gradient, and relavent constant
    gL = Dt(sigma) + A(delta,2);
    % gradient, divided by beta
    g = beta*gD + mu*gA - gL;

end

out.total_iter = numel(out.objf_val)-1;
out.final_error = norm(A(U(:),1)-b)/nrmb;
% output these so one may check optimality conditions
out.optimallity = sigma + beta*(W-Uc);
out.optimallity2 = mu*gA + beta*gD - gL;
out.elapsed_time = toc;

fprintf('total iteration count: %i\n',out.total_iter);
fprintf('||Au-b||/||b||: %5.3f\n',out.final_error);
fprintf('mu/2*||Au-b||^2 + ||Du||_1: %g\n',out.objf_val(end));


function W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind)

if strcmp(L1type,'anisotropic')
    W = Uc - sigma/beta;
    W = max(abs(W) - 1/beta, 0).*sign(W);    
elseif strcmp(L1type,'isotropic')
    W = Uc - sigma/beta;
    Ucbar_norm = sqrt(W(:,1).*conj(W(:,1)) + ...
        W(:,2).*conj(W(:,2)) + W(:,3).*conj(W(:,3)));
    Ucbar_norm = max(Ucbar_norm - 1/beta, 0)./(Ucbar_norm+eps);
    W(:,1) = W(:,1).*Ucbar_norm;
    W(:,2) = W(:,2).*Ucbar_norm;
    W(:,3) = W(:,3).*Ucbar_norm;
else
    error('Somethings gone wrong.  L1type is either isotropic or anisotropic');
end

% reset edge values if not using periodic regularization
if ~wrap_shrink, W(ind)=Uc(ind); end


