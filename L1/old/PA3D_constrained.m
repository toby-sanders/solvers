function [U, out] = PA3D_constrained(A,b,p,q,r,opts)

% Modifications by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016


% This code has been modified to solve l1 penalty problems with the
% polynomial annihilation transform.  Several small bugs and notation
% changes have been made as well.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original motivation to find:

%               min_f { mu/2 ||Af - b||_2^2 + ||D^k f||_1 }

% where D^k is kth order finite difference
% The problem is modified using variable splitting
% and this algorithm solves: 

%      min_{f,w} {mu/2 ||Af - b||_2^2 + beta/2 ||D^k f - w ||_2^2 
%               ||w||_1 - (delta , Af - b ) - (sigma , D^k f - w) }

% delta and sigma are Lagrange multipliers
% Algorithm uses alternating direction minimization over f and w.



% This algorithm was originally authored by Chengbo Li at Rice University.
% original code and description can be found here: 
% http://www.caam.rice.edu/~optimization/L1/TVAL3/

% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   b: data values in vector form
%   p,q,r: signal dimensions
%   opts: structer containing opts, see function check_PA_opts.m for these


% Outputs:
%   U: reconstructed signal
%   out: output numerics





% get and check opts
opts = check_PA_opts(opts);


% mark important constants
tol_inn = opts.tol_inn;
tol_out = opts.tol_out;
k = opts.order;
n = p*q*r;
wrap_shrink = opts.wrap_shrink;
if round(k)~=k
    wrap_shrink = true;
end



% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end

% check scaling A
if opts.scale_A
    [A,b] = ScaleA(n,A,b);
end

% check scaling b
scl = 1;
if opts.scale_b
    [b,scl] = Scaleb(b);
end

% check for maximum constraint value
if opts.max_c
    max_v = opts.max_v*scl;
end


% calculate A'*b
Atb = A(b,2);


% initialize everything else
global D Dt
[U,mu,beta,muf,betaf,beta2,sigma,delta,xi,gL,ind,out] ...
    = get_init_const(p,q,r,Atb,scl,opts,k,b,wrap_shrink);    % U: p*q

nrmb = norm(b);
Upout = U;
Uc = D(U);


max_v = 1*scl;
min_v = 0;


% first shrinkage step
W = max(abs(Uc) - 1/beta, 0).*sign(Uc);
% reset edge values if not using periodic regularization
if ~wrap_shrink, W(ind)=Uc(ind); end

Z = max(min(U(:),max_v),min_v);

lam1 = sum(sum(sum(abs(W))));

% gA and gD are the gradients of ||Au-b||^2 and ||Du-w||^2, respectively
% i.e. g = A'(Au-b), gD = D'(Du-w)
[lam2,lam3,lam4,lam5,lam6,lam7,f,gD,Au,gA,gZ] = get_grad(U,Uc,W,Z,...
    lam1,beta,mu,beta2,A,b,Atb,sigma,delta,xi);


% compute gradient
g = beta*gD + mu*gA + beta2*gZ - gL;


out.f = [out.f; f]; 
out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; 
out.lam3 = [out.lam3; lam3];out.lam4 = [out.lam4; lam4]; 
out.lam5 = [out.lam5; lam5];out.mu = [out.mu; mu];
out.lam6 = [out.lam6; lam6];out.lam7 = [out.lam7; lam7];
out.DU = [out.DU;norm(Uc(:),1)];


for ii = 1:opts.outer_iter
    if opts.disp
            fprintf('    Beginning outer iteration #%d\n',ii);
            fprintf('    mu = %d , beta = %d , order = %g\n',mu,beta,k);
            fprintf('iter    ||w||_1    ||Du - w||^2  ||Au - b||^2   ||Du||_1\n');
    end
        
    %initialize the constants
    gam = opts.gam; Q = 1; fp = f;
    
    
    for jj = 1:opts.inner_iter
        % compute step length, tau
        if jj~=1
            % BB-like step length
            dgA = gA - gAp;   
            dgD = gD - gDp;
            dgZ = gZ - gZp;
            ss = uup'*uup;                      
            sy = uup'*(beta*dgD + mu*dgA + beta2*dgZ);       
            tau = abs(ss/max(sy,eps));          
        else
            % do Steepest Descent at the 1st ieration
            Dg = D(reshape(g,p,q,r));       
            dDg = sum(sum(sum(Dg.*conj(Dg))));
            Ag = A(g,1);
            tau = abs((g'*g)/(beta*dDg + mu*(Ag')*Ag));
        end

        % keep previous values for backtracking & computing next tau
        Up = U; gAp = gA; gDp = gD; gZp = gZ; Aup = Au; 
        Ucp = Uc; %DtsAtdp =  DtsAtd;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ONE-STEP GRADIENT DESCENT %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = U(:) - tau*g;
        % projected gradient method for inequality constraints
        %if opts.nonneg
        %    U = max(real(U),0);
        %elseif opts.isreal
        %    U = real(U);
        %end
        %if opts.max_c
        %    U = min(U,max_v);
        %end
        U = reshape(U,p,q,r);
        Uc = D(U);

        [lam2,lam3,lam4,lam5,lam6,lam7,f,gD,Au,gA,gZ] = get_grad(U,Uc,W,Z,...
            lam1,beta,mu,beta2,A,b,Atb,sigma,delta,xi);

        % Nonmonotone Line Search Back tracking
        % Unew = Up + alpha*(U - Up)
        % f should be decreasing, if not, then the algorithm moves U
        % back in the direction of the previous solution
        alpha = 1;
        du = U - Up;
        const = 1e-5*beta*(g'*g*tau);
        cnt = 0; flg = true;
        
        


        % if back tracking is successful, then recompute
        if alpha ~= 0
            Ucbar = Uc - sigma/beta;
            W = max(abs(Ucbar) - 1/beta, 0).*sign(Ucbar);
            % reset edge values if not using periodic regularization
            if ~wrap_shrink, W(ind)=Uc(ind); end
            % update parameters related to Wx, Wy
            [lam1,lam2,lam4,f,gD] = update_W(beta,...
                W,Uc,sigma,lam1,lam2,lam4,f);
            Uzbar = U(:) - xi/beta2;
            Z = max(min(Uzbar,max_v),min_v);
            [lam6,lam7,f,gZ] = update_Z(beta2,...
                Z,U,xi,lam6,lam7,f);
        end

        % update reference value
        Qp = Q; Q = gam*Qp + 1; fp = (gam*Qp*fp + f)/Q;
        uup = U - Up; uup = uup(:);           % uup: pqr
        rel_chg_inn = norm(uup)/norm(Up(:));
        
        
        
        out.f = [out.f; f]; out.C = [out.C; fp]; out.cnt = [out.cnt;cnt];
        out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; out.lam3 = [out.lam3; lam3];
        out.lam4 = [out.lam4; lam4]; out.lam5 = [out.lam5; lam5];
        out.lam7 = [out.lam7; lam7]; out.lam6 = [out.lam6; lam6];
        out.tau = [out.tau; tau]; out.alpha = [out.alpha; alpha];out.mu = [out.mu; mu];
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        out.rel_lam2 = [out.rel_lam2;sqrt(lam2)/norm(W(:))];
        out.DU = [out.DU; norm(Uc(:),1)];
        if opts.store_soln
            out.Uall(:,:,jj+(ii-1)*opts.inner_iter) = U;
        end

        
        
        
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            prnt_format = '%3.0f %10.5g %12.5g %13.5g %10.5g\n';
            fprintf(prnt_format, jj,lam1,lam2,lam3,out.DU(end));
        end
       


        % recompute gradient
        g = beta*gD + mu*gA + beta2*gZ - gL;
        
        % move to next outer iteration and update multipliers if relative
        % change is less than tolerance
        if (rel_chg_inn < tol_inn), break; end;
        
        
    end
    % end of inner loop
    
    
    rel_chg_out = norm(U(:)-Upout(:))/norm(Upout(:));
    out.rel_chg_out = [out.rel_chg_out; rel_chg_out];
    Upout = U;

    % stop if already reached optimal solution
    if rel_chg_out < tol_out || sqrt(lam3(end))/nrmb<opts.min_l2_error
        break;
    end

    % update multipliers
    [sigma,delta,xi,lam4,lam5,lam7] = update_mlp(beta,mu,beta2, ...
        W,Uc,gZ,Au,b,sigma,delta,xi);
    if ~opts.data_mlp, delta(:) = 0; lam5 = 0; end


    % update penality parameters for continuation scheme
    %beta0 = beta;
    beta = min(betaf, beta*opts.rate_ctn);
    mu = min(muf, mu*opts.rate_ctn);
    beta2 = beta;

    % update function value, gradient, and relavent constant
    f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5 + beta2/2*lam6 - lam7;
    %gL = -(beta0/beta)*g;     % DtsAtd should be divided by new beta  
    gL = (Dt(sigma) + A(delta,2) + xi);
    % gradient
    g = beta*gD + mu*gA + beta2*gZ - gL;

end


out.total_iter = jj*ii;
out.final_error = norm(A(U(:),1)-b)/nrmb;
out.final_wl1 = lam1(end);
out.final_Du_w = lam2(end);
out.rel_error = sqrt(out.lam3)/nrmb;
if out.rel_error(end) < opts.min_l2_error
    fprintf('\nREACHED OPTIMAL L2 ERROR!!!\n\n');
end

final_disp(out,opts);
            
% rescale U
U = U/scl;




function [lam2,lam3,lam4,lam5,lam6,lam7,f,gD,Au,gA,gZ] = get_grad(U,Uc,W,Z,...
    lam1,beta,mu,beta2,A,b,Atb,sigma,delta,xi)
global Dt

Au = A(U(:),1);

% gA = A'(Au-b)
gA = A(Au,2) - Atb;

gZ = U(:)-Z;

% lam2, ||Du - w||^2
V = Uc - W;
lam2 = sum(sum(sum(V.*conj(V))));

% gD = D'(Du-w)
gD = Dt(V);

% lam3, ||Au - b||^2
Aub = Au-b;
lam3 = Aub'*Aub;%norm(Aub)^2;

%lam4
lam4 = sum(sum(sum(sigma.*V)));

%lam5
lam5 = delta'*Aub;

%lam6
lam6 = gZ'*gZ;

%lam7
lam7 = xi'*gZ;

% f
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5 + beta2/2*lam6 - lam7;



function [U,lam2,lam3,lam4,lam5,f,Uc,Au,gA,gD,gZ] = back_up(p,q,r,lam1,...
    alpha,beta,mu,Up,du,gAp,dgA,gDp,dgD,gZp,dgZ,Aup,dAu,W,Ucp,dUc,...
    b,sigma,delta,xi)

gA = gAp + alpha*dgA;
gD = gDp + alpha*dgD;
gZ = gZp + alpha*dgZ;
U = Up + alpha*reshape(du,p,q,r);
Au = Aup + alpha*dAu;
Uc = Ucp + alpha*dUc;


V = Uc - W;


lam2 = sum(sum(sum(V.*conj(V))));
Aub = Au-b;
lam3 = norm(Aub)^2;
lam4 = sum(sum(sum(sigma.*V)));
lam5 = delta'*Aub;
lam6 = gZ'*gZ;
lam7 = xi'.*gZ;
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5 + beta2/2*lam6 - lam7;



function [lam1,lam2,lam4,f,gD] = update_W(beta,...
    W,Uc,sigma,lam1,lam2,lam4,f)
global Dt

% update parameters because W was updated
tmpf = f -lam1 - beta/2*lam2 + lam4;
lam1 = sum(sum(sum(abs(W))));
V = Uc - W;

gD = Dt(V);
lam2 = sum(sum(sum(V.*conj(V))));
lam4 = sum(sum(sum(sigma.*V)));
f = tmpf +lam1 + beta/2*lam2 - lam4;

function [lam6,lam7,f,gZ] = update_Z(beta2,...
    Z,U,xi,lam6,lam7,f)

tmpf = f - beta2/2*lam6 + lam7;
gZ = U(:) - Z;
lam6 = gZ'*gZ;
lam7 = xi'*gZ;
f = tmpf + beta2/2*lam6 - lam7;


function [sigma,delta,xi,lam4,lam5,lam7] = update_mlp(beta,mu,beta2, ...
    W,Uc,gZ,Au,b,sigma,delta,xi)


V = Uc - W;
sigma = sigma - beta*V;
Aub = Au-b;
delta = delta - mu*Aub;
xi = xi - beta2*gZ;


%tmpf = f + lam4 + lam5;
lam4 = sum(sum(sum(sigma.*V)));
lam5 = delta'*Aub;
lam7 = xi'*gZ;
%f = tmpf - lam4 - lam5;





