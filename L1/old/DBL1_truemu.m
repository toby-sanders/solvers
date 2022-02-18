function [U, out] = DBL1_truemu(A,b,n,opts,Utrue)


% Modifications by Toby Sanders @ASU
% School of Math & Stat Sciences
% Last update: 01/08/2017
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
% Algorithm uses alternating direction minimization over f and w.
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

% constants
%   lam1: ||w||_1
%   lam2: ||Du-w||^2
%   lam3: ||Au-b||^2
%   lam4: (sigma,Du-w)
%   lam5: (delta,Au-b) , default for this is delta = 0.

if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3
    error('n can have at most 3 dimensions');
end
p = n(1); q = n(2); r = n(3);

% get and check opts
opts = check_HOTV_opts(opts);

% mark important constants
tol = opts.tol; tol_inn = 1e-4;
k = opts.order; n = p*q*r;

% check that A* is true adjoint of A
% check scaling of parameters, maximum constraint value, etc.
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end; clear flg;
if opts.scale_A, [A,b] = ScaleA(n,A,b); end
[~,scl] = Scaleb(b);
if opts.scale_mu, opts.mu = opts.mu*scl; opts.stmu = opts.stmu*scl; end
opts.beta = opts.beta*scl;

% initialize everything else
Atb = A(b,2); % A'*b
[U,mu,beta,~,~,muDbeta,~,delta,gL,ind,out] ...
    = get_HOTV(p,q,r,Atb,A(Atb,1),scl,opts,k,b,true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Changes for the wavelets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global D Dt
if q==1, [D,Dt] = my_wav_trans_1D(['db',num2str(k)],opts.levels);
else, [D,Dt] = my_wavelet_trans_2D(['db',num2str(k)],opts.levels,p,q);
end
wrap_shrink = true; L1type = 'anisotropic';
sigma = D(zeros(p,q,r)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nrmb = norm(b);
Uc = D(U);

% first shrinkage step
[W,lam1] = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,[],[],[],[],ind,0);


% gA and gD are the gradients w.r.t. u of ||Au-b||^2 and ||Du-w||^2
[lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
    lam1,beta,mu,A,b,Atb,sigma,delta);
g = gD + muDbeta*gA - gL; % full obj gradient


out.f = [out.f; f]; 
out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; 
out.lam3 = [out.lam3; lam3];out.lam4 = [out.lam4; lam4]; 
out.lam5 = [out.lam5; lam5];out.mu = [out.mu; mu];
out.err_true =1; 
stmu = opts.stmu; % input some step length to increase mu
ii = 0;
while numel(out.f) <= opts.max_iter
    ii = ii + 1;
    % do Steepest Descent at the 1st iteration
    gc = D(reshape(g,p,q,r));       
    dDd = sum(col(gc.*conj(gc)));
    Ag = A(g,1);
    tau = abs((g'*g)/(dDd + muDbeta*(Ag')*Ag));
    % check for convergence after the Lagrange multiplier updates
    if ii~=1
        rel_chg_out = norm(tau*g)/norm(U(:));
        out.rel_chg_out = [out.rel_chg_out; rel_chg_out];
        if rel_chg_out < tol, break; end
    end
    % reset constants
    gam = opts.gam; Q = 1; fp = f;
    
    for jj = 1:opts.inner_iter
        % compute step length, tau
        if jj~=1
            % BB-like step length
            dgA = gA - gAp;   
            dgD = gD - gDp;                    
            ss = uup'*uup;                      
            sy = uup'*(dgD + muDbeta*dgA);       
            tau = abs(ss/max(sy,eps));   
        end

        % keep previous values for backtracking & computing next tau
        Up = U; gAp = gA; gDp = gD; Aup = Au; 
        Ucp = Uc; %DtsAtdp =  DtsAtd;

        % gradient decent
        U = U(:) - tau*g;
        
        % projected gradient method for inequality constraints
        if opts.nonneg, U = max(real(U),0);
        elseif opts.isreal, U = real(U); end
        if opts.max_c, U = min(U,opts.max_v); end
        U = reshape(U,p,q,r);  Uc = D(U);
        
        [lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
            lam1,beta,mu,A,b,Atb,sigma,delta);

        % Nonmonotone Line Search Back tracking
        % Unew = Up + alpha*(U - Up)
        alpha = 1;
        du = U - Up;
        const = 1e-5*beta*(g'*g*tau);
        cnt = 0; flg = true;       
        while f > fp - alpha*const
            if cnt <5
                if flg
                    dgA = gA - gAp;
                    dgD = gD - gDp;
                    dAu = Au - Aup;
                    dUc = Uc - Ucp;
                    flg = false;
                end
                % shrink alpha
                alpha = alpha*opts.gamma; 
                % U = alpha*U +(1-alpha)Up;
                [U,lam2,lam3,lam4,lam5,f,Uc,Au,gA,gD] = back_up(p,q,r,...
                    lam1,alpha,beta,mu,Up,du,gAp,dgA,gDp,dgD,Aup,dAu,W,...
                    Ucp,dUc,b,sigma,delta);
                cnt = cnt + 1;
            else               
                % shrink gam
                gam = opts.rate_gam*gam;

                % give up and take Steepest Descent step
                if (opts.disp > 0) && (mod(jj,opts.disp) == 0)
                    disp(' count of back tracking attains 5, taking steepest decent');
                end

                % compute step length, tau
                gc = D(reshape(g,p,q,r));
                dDd = sum(col(gc.*conj(gc)));
                Ag = A(g,1);
                tau = abs((g'*g)/(dDd + muDbeta*(Ag')*Ag));
                %update
                U = Up(:) - tau*g;
                % projected gradient method for inequality constraints
                if opts.nonneg, U = max(real(U),0);
                elseif opts.isreal, U = real(U); end
                if opts.max_c, U = min(U,opts.max_v); end
                
                U = reshape(U,p,q,r);
                Uc = D(U);
                % shrinkage
                [W,lam1] = shrinkage(Uc,L1type,...
                    beta,sigma,wrap_shrink,[],lam2,lam4,f,ind,0);
                
                [lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
                    lam1,beta,mu,A,b,Atb,sigma,delta);
                alpha = 0; % remark the failure of back tracking
                break;
            end
            
        end
        

        % if back tracking is successful, then recompute
        if alpha ~= 0
            [W,lam1,lam2,lam4,f,gD] = shrinkage(Uc,L1type,...
                beta,sigma,wrap_shrink,lam1,lam2,lam4,f,ind,1);
        end

        % update reference value
        Qp = Q; Q = gam*Qp + 1; fp = (gam*Qp*fp + f)/Q;
        uup = U - Up; uup = uup(:);           % uup: pqr
        rel_chg_inn = norm(uup)/norm(Up(:));
        
        out.f = [out.f; f]; out.C = [out.C; fp]; out.cnt = [out.cnt;cnt];
        out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; 
        out.lam3 = [out.lam3; lam3];out.lam4 = [out.lam4; lam4]; 
        out.lam5 = [out.lam5; lam5]; out.tau = [out.tau; tau]; 
        out.alpha = [out.alpha; alpha];out.mu = [out.mu; mu];
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        %out.rel_lam2 = [out.rel_lam2;sqrt(lam2)/norm(W(:))];
        %out.DU = [out.DU; norm(Uc(:),1)];
        if opts.store_soln
            out.Uall(:,:,jj+(ii-1)*opts.inner_iter) = U;
        end
      
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            fprintf('iter=%i, rel_chg_sol=%10.4e, ||Du-w||^2=%10.4e\n',...
                numel(out.f)-1, rel_chg_inn, lam2);
        end
        % recompute gradient
        g = gD + muDbeta*gA - gL;
        % check inner loop convergence
        if (rel_chg_inn < tol_inn) || numel(out.f)>opts.max_iter, break; end 

    end
    % end of inner loop
    % update mu
    if ii>10
        out.err_true = [out.err_true;myrel(U,Utrue,2)];
        fprintf('true error = %g, mu = %g\n',out.err_true(end),mu);
        % if error increases, switch direction of stmu
        if out.err_true(end)>out.err_true(end-1)
            if abs(stmu)<.5
                out.mu_optimal = mu -stmu;
            out.err_optimal = out.err_true(end-1);
            U = Upout;
            break;
            end
            stmu = -stmu/4;           
        end
        mu = mu+stmu;
        if mu<=0
            mu = mu - stmu;
            stmu = stmu/4;
            mu = mu + stmu;
        end
    end
    muDbeta = mu/beta;
    Upout = U;
    
    % update multipliers
    deltap = delta;
    lam5p = lam5;
    [sigma,delta,lam4,lam5] = update_mlp(beta,mu, ...
        W,Uc,Au,b,sigma,delta);
    if ii>=opts.data_mlp, delta = deltap; lam5 = lam5p;  end
    if opts.disp, fprintf(' Lagrange mlps update number %i\n',ii);end
    
    % update function value, gradient, and relavent constant
    f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;
    gL = 1/beta*(Dt(sigma) + A(delta,2));
    % gradient, divided by beta
    g = gD + muDbeta*gA - gL;

end

out.total_iter = numel(out.f)-1;
out.final_error = norm(A(U(:),1)-b)/nrmb;out.final_wl1 = lam1(end);
out.final_Du_w = lam2(end); out.rel_error = sqrt(out.lam3)/nrmb;
out.optimallity = sigma + beta*(W-Uc); out.W = W;
out.optimallity2 = muDbeta*gA + gD - gL;

final_disp(out,opts);
if opts.disp_fig && opts.isreal
    figure(132);
    if r==1 && q ==1, plot((U));
    elseif r==1, imagesc((U));
    end
end




function [lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
    lam1,beta,mu,A,b,Atb,sigma,delta)
global Dt

Au = A(U(:),1);
gA = A(Au,2) - Atb;% gA = A'(Au-b)

% lam2, ||Du - w||^2
V = Uc - W;
lam2 = sum(col(V.*conj(V)));
gD = Dt(V);

% update obj function values
Aub = Au-b;
lam3 = Aub'*Aub;
lam4 = sum(col(sigma.*V));
lam5 = delta'*Aub;
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;



function [U,lam2,lam3,lam4,lam5,f,Uc,Au,gA,gD] = back_up(p,q,r,lam1,...
    alpha,beta,mu,Up,du,gAp,dgA,gDp,dgD,Aup,dAu,W,Ucp,dUc,...
    b,sigma,delta)

gA = gAp + alpha*dgA;
gD = gDp + alpha*dgD;
U = Up + alpha*reshape(du,p,q,r);
Au = Aup + alpha*dAu;
Uc = Ucp + alpha*dUc;
V = Uc - W;

lam2 = sum(col(V.*conj(V)));
Aub = Au-b;
lam3 = norm(Aub)^2;
lam4 = sum(col(sigma.*V));
lam5 = delta'*Aub;
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;


function [sigma,delta,lam4,lam5] = update_mlp(beta,mu, ...
    W,Uc,Au,b,sigma,delta)

V = Uc - W;
sigma = sigma - beta*V;
Aub = Au-b;
delta = delta - mu*Aub;

lam4 = sum(col(sigma.*V));
lam5 = delta'*Aub;


function [W,lam1,lam2,lam4,f,gD] = shrinkage(Uc,L1type,...
    beta,sigma,wrap_shrink,lam1,lam2,lam4,f,ind,flg)

global Dt

if flg
    tmpf = f -lam1 - beta/2*lam2 + lam4;
end

if strcmp(L1type,'anisotropic')
    W = Uc - sigma/beta;
    W = max(abs(W) - 1/beta, 0).*sign(W);    
    lam1 = sum(col(abs(W)));
elseif strcmp(L1type,'isotropic')
    W = Uc - sigma/beta;
    Ucbar_norm = sqrt(W(:,1).*conj(W(:,1)) + ...
        W(:,2).*conj(W(:,2)) + W(:,3).*conj(W(:,3)));
    Ucbar_norm = max(Ucbar_norm - 1/beta, 0)./(Ucbar_norm+eps);
    W(:,1) = W(:,1).*Ucbar_norm;
    W(:,2) = W(:,2).*Ucbar_norm;
    W(:,3) = W(:,3).*Ucbar_norm;
    lam1 = sum(sqrt(W(:,1).*conj(W(:,1)) + ...
        W(:,2).*conj(W(:,2)) + W(:,3).*conj(W(:,3))));
else
    error('Somethings gone wrong.  L1type is either isotropic or anisotropic');
end

% reset edge values if not using periodic regularization
if ~wrap_shrink, W(ind)=Uc(ind); end

if flg
    % update parameters because W was updated
    V = Uc - W;
    gD = Dt(V);
    lam2 = sum(col(V.*conj(V)));
    lam4 = sum(col(sigma.*V));
    f = tmpf +lam1 + beta/2*lam2 - lam4;
end
