function [U, out] = HOTV3D_POIS(A,b,n,opts)


% Modifications by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/17/2017
%
% This version of the algorithm is a modification of my general L1
% optimization algorithm to account for Poisson noise.  It accounts for
% the Poisson noise statistics by estimating variances that lead to 
% weights, R, for a reweighted least squares data fitting.  

% *************** VERY IMPORTANT (AND USEFUL)  ******************
% The important parameter mu is automatically apdapted using the 
% "discrepancy principle" (the noise variance in the data should match the 
% means), i.e. E[ ||Au-b||^2 ] =  sum(Au) 
% HOWEVER, THIS WILL ONLY WORK PROPERLY IF THE DATA IS TRUELY POISSON AND 
% HAS NOT BEEN RESCALED IN ANY MANNER.
%  **************************************************************
%
%
%
% This code has been modified to solve l1 penalty problems with
% higher order TV operators.  Several small bugs and notation
% changes have been made as well.  Aside from notation and reorganization
% of the code, we list some important changes below.
% 1. The algorithm solves TV and higher order (HO) TV methods, as well as
% the more general multiscale HOTV.  These parameters are specfied by the
% opts fields 'order' and 'levels'.
% 2. For TV And HOTV this algoirthm can solve the isotropic and anisotropic
% variants of the TV norms.  Isotropic is generally preferred. This
% parameter is specified by the field 'L1type'.
% 3. The algorithm can recover complex signals.  Some inner products were
% incorrectly computed without conjugations.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [U, out] = HOTV3D(A,b,n,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||R(Au - b)||_2^2 + ||D^k u||_1 }
%
% where D^k is kth order finite difference, and R is a diagonal matrix with
% weights estimated by R_{ii} = (sqrt((Au)_i) + epsilon)^-1.

% Multiscale finite differences D^k can also be used.
% To see how to modify these settings read the file "check_HOTV_opts.m"
%
% The problem is modified using variable splitting
% and this algorithm solves: 
%
%      min_{u,w} {mu/2 ||R(Au - b)||_2^2 + beta/2 ||D^k u - w ||_2^2 
%               + ||w||_1 - (delta , Au - b ) - (sigma , D^k u - w) }
%
% delta and sigma are Lagrange multipliers
% Algorithm uses alternating direction minimization over f and w.
%
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




% Lambdas
%   lam1: ||w||_1
%   lam2: ||Du-w||^2
%   lam3: ||Au-b||^2
%   lam4: (sigma,Du-w)
%   lam5: (delta,Au-b)  , default for this is delta = 0.



    


if numel(n)<3
    n(end+1:3) = 1;
elseif numel(n)>3
    error('n can have at most 3 dimensions');
end
p = n(1); q = n(2); r = n(3);



% get and check opts
opts = check_HOTV_opts(opts);


% mark important constants
tol_inn = opts.tol_inn;
tol_out = opts.tol_out;
mu = opts.mu;
scl = 1; % scl
k = opts.order;
n = p*q*r;
wrap_shrink = opts.wrap_shrink;
L1type = opts.L1type;
rel_chg_out = 0;
stepmu = .2;
flg_inc = 1;
flg_dec = 1;
max_mu = 1e5;
min_mu = 1e-6;
epsilon = .1;
if round(k)~=k || opts.levels>1
    wrap_shrink = true; % indices aren't set up for these cases,�\_(?)_/� so just do shrinkage everywhere
end



% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end
%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);

if ~flg
    error('A and A* do not appear consistent');
end
clear flg;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is for Poisson noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sum(b<0)
    warning('data vector b should be nonnegative');
end
% create scaled b for weights
btmp = b/max(b);
R = 1./(sqrt(btmp)+epsilon); % proper weights assuming poisson noise
R = R/max(R);   % rescale them
% save the original operator and original b for later
AO = A; 
borig = b;  

% weighted operator and weighted b
A = @(u,mode)my_local_oper(AO,R,u,mode);
b = b.*R;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







% check for maximum constraint value
if opts.max_c
    max_v = opts.max_v*scl;
end


% calculate A'*b
Atb = A(b,2);


% initialize everything else
global D Dt
[U,~,beta,~,betaf,muDbeta,sigma,delta,gL,ind,out] ...
    = get_HOTV(p,q,r,Atb,A(Atb,1),scl,opts,k,b,wrap_shrink);    % U: p*q
nrmb = norm(b);
Upout = U;
Uc = D(U);



% first shrinkage step
[W,lam1] = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,[],[],[],[],ind,0);


% gA and gD are the gradients of ||Au-b||^2 and ||Du-w||^2, respectively
% i.e. g = A'(Au-b), gD = D'(Du-w)
[lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
    lam1,beta,mu,A,b,Atb,sigma,delta);


% compute gradient
g = gD + muDbeta*gA - gL;


out.f = [out.f; f]; 
out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; 
out.lam3 = [out.lam3; lam3];out.lam4 = [out.lam4; lam4]; 
out.lam5 = [out.lam5; lam5];out.mu = [out.mu; mu];
out.r1 = [];out.sigma = [];
out.R = [];out.var = [];out.means = [];


for ii = 1:opts.outer_iter
    if opts.disp
            fprintf('    Beginning outer iteration #%d\n',ii);
            fprintf('    mu = %d , beta = %d , order = %g, rel chg =%g\n',mu,beta,k,rel_chg_out);
            fprintf('iter    ||w||_1    ||Du - w||^2  ||Au - b||^2   rel chg\n');
    end
        
    %initialize the constants
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
        else
            % do Steepest Descent at the 1st ieration
            gc = D(reshape(g,p,q,r));       
            dDd = sum(col(gc.*conj(gc)));
            Ag = A(g,1);
            tau = abs((g'*g)/(dDd + muDbeta*(Ag')*Ag));
        end

        % keep previous values for backtracking & computing next tau
        Up = U; gAp = gA; gDp = gD; Aup = Au; 
        Ucp = Uc; %DtsAtdp =  DtsAtd;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ONE-STEP GRADIENT DESCENT %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = U(:) - tau*g;
        
        % projected gradient method for inequality constraints
        if opts.nonneg
            U = max(real(U),0);
        elseif opts.isreal
            U = real(U);
        end
        if opts.max_c
            U = min(U,max_v);
        end
        %figure(122);plot(U);%pause;
        U = reshape(U,p,q,r);
        Uc = D(U);
        

        [lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
            lam1,beta,mu,A,b,Atb,sigma,delta);

        % Nonmonotone Line Search Back tracking
        % Unew = Up + alpha*(U - Up)
        % f should be decreasing, if not, then the algorithm moves U
        % back in the direction of the previous solution
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
                % U is moved back toward Up, in particular: 
                % U = alpha*U +(1-alpha)Up;
                % all other values are updated accordingly
                [U,lam2,lam3,lam4,lam5,f,Uc,Au,gA,gD] = back_up(p,q,r,...
                    lam1,alpha,beta,mu,Up,du,gAp,dgA,gDp,dgD,Aup,dAu,W,...
                    Ucp,dUc,b,sigma,delta);
                cnt = cnt + 1;
            else
                
                % shrink gam
                gam = opts.rate_gam*gam;

                % give up and take Steepest Descent step
                if (opts.disp > 0) && (mod(jj,opts.disp) == 0)
                    disp('    count of back tracking attains 5 ');
                end

                % compute step length, tau
                gc = D(reshape(g,p,q,r));
                dDd = sum(col(gc.*conj(gc)));
                Ag = A(g,1);
                tau = abs((g'*g)/(dDd + muDbeta*(Ag')*Ag));
                %update
                U = Up(:) - tau*g;
                % projected gradient method for inequality constraints
                if opts.nonneg
                    U = max(real(U),0);
                elseif opts.isreal
                    U = real(U);
                end
                
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
        out.lam3 = [out.lam3; lam3];
        out.lam4 = [out.lam4; lam4]; out.lam5 = [out.lam5; lam5];
        out.tau = [out.tau; tau]; out.alpha = [out.alpha; alpha];
        %out.mu = [out.mu; mu];
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        if opts.store_soln
            out.Uall(:,:,jj+(ii-1)*opts.inner_iter) = U;
        end

        
        
        
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            prnt_format = '%3.0f %10.5g %12.5g %13.5g %10.5f\n';
            fprintf(prnt_format, jj,lam1,lam2,lam3,rel_chg_inn);
        end


        % recompute gradient
        g = gD + muDbeta*gA - gL;
        
        % move to next outer iteration and update multipliers if relative
        % change is less than tolerance
        if (rel_chg_inn < tol_inn), break; end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This section is for Poisson noise
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if mod(jj,5)==0
            % update weights for Poisson noise
            Rp = R;
            R = AO(U(:),1);    % new estimate for b
            %[btmp,~] = Scaleb(btmpo);
            R = R/max(R);
            R = 1./(sqrt(R)+epsilon); % weights
            R = R/max(R);   % rescale them
            out.R = [out.R;myrel(R,Rp)];
            
            % new weighted operator and weighted b and A'*b
            A = @(u,mode)my_local_oper(AO,R,u,mode);
            b = borig.*R;
            Atb = A(b,2);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    % end of inner loop
    
    
    rel_chg_out = norm(U(:)-Upout(:))/norm(Upout(:));
    out.rel_chg_out = [out.rel_chg_out; rel_chg_out];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is for Poisson noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % discrepancy principle
    if mod(ii,5)==0      
        [mu,flg_inc,flg_dec,out] ...
            = get_mu(AO(U(:),1),borig,mu,stepmu,max_mu,min_mu,flg_dec,flg_inc,out);
    end
    
    Upout = U;

    % stop if already reached optimal solution
    if rel_chg_out < tol_out || sqrt(lam3(end))/nrmb<opts.min_l2_error
        break;
    end
    
    
    
    % update multipliers
    deltap = delta;
    lam5p = lam5;
    [sigma,delta,lam4,lam5] = update_mlp(beta,mu, ...
        W,Uc,Au,b,sigma,delta);
    muDbeta = mu/beta;
    if ii>=opts.data_mlp, delta = deltap; lam5 = lam5p;  end
    

    


    % update penality parameters for continuation scheme
    beta = min(betaf, beta*opts.rate_ctn);
    

    % update function value, gradient, and relavent constant
    f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;
    %gL = -(beta0/beta)*g;     % DtsAtd should be divided by new beta  
    gL = 1/beta*(Dt(sigma) + A(delta,2));
    % gradient, divided by beta
    g = gD + muDbeta*gA - gL;

end

out.total_iter = numel(out.f)-1;
out.final_error = norm(A(U(:),1)-b)/nrmb;
out.final_wl1 = lam1(end);
out.final_Du_w = lam2(end);
out.rel_error = sqrt(out.lam3)/nrmb;
if out.rel_error(end) < opts.min_l2_error
    fprintf('\nREACHED OPTIMAL L2 ERROR!!!\n\n');
end

final_disp(out,opts);
if opts.disp_fig && opts.isreal
    if r==1 & q ==1
        figure(132);
        plot((U));
    elseif r==1
        figure(132);
        imagesc((U));
    end
end

            
% rescale U
U = U/scl;






function [lam2,lam3,lam4,lam5,f,gD,Au,gA] = get_grad(U,Uc,W,...
    lam1,beta,mu,A,b,Atb,sigma,delta)
global Dt

Au = A(U(:),1);

% gA = A'(Au-b)
gA = A(Au,2) - Atb;

% lam2, ||Du - w||^2
V = Uc - W;
lam2 = sum(col(V.*conj(V)));

% gD = D'(Du-w)
gD = Dt(V);

% lam3, ||Au - b||^2
Aub = Au-b;
lam3 = Aub'*Aub;%norm(Aub)^2;

%lam4
lam4 = sum(col(sigma.*V));

%lam5
lam5 = delta'*Aub;

% f
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

%tmpf = f + lam4 + lam5;
lam4 = sum(col(sigma.*V));
lam5 = delta'*Aub;
%f = tmpf - lam4 - lam5;




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



function x = my_local_oper(A,R,x,mode)

switch mode
    case 1
        x = A(x,1);
        x = R.*x;
    case 2
        x = R.*x;
        x = A(x,2);
end


function [mu,flg_inc,flg_dec,out] ...
    = get_mu(Ax,b,mu,stepmu,max_mu,min_mu,flg_dec,flg_inc,out)
% We check the condition that the mean is approximately the variance
% i.e.  E[ ||Ax-b||^2 ] =  sum(Ax), for Poisson noise.
% mu is updated accordingly

% compute relevant terms
r1 = Ax-b;
r1 = r1'*r1;
r2 = sum(Ax);

out.var = [out.var;r1];
out.means = [out.means;r2];
% check if condition is approximately satisfied
if abs(1 - r1/r2)>5e-2
   r1 = 1-r1/r2; % compute signed difference to determine change needed
   if r1 < 0  % increase mu
       if flg_inc == 0
           stepmu = stepmu*.5;
       end
       flg_inc = 1; % these flags just keep track of the direction of mu
       flg_dec = 0;
       mu = min(mu*(1 + stepmu),max_mu);
   else  % decrease mu
       if flg_dec == 0
           stepmu = stepmu*.5;
       end
       flg_dec = 1;
       flg_inc = 0;
       mu = max(mu*(1-stepmu),min_mu);
   end

end
out.r1 = [out.r1;r1];
out.mu = [out.mu;mu];
