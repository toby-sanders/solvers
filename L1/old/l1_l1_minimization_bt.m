function [U, out] = l1_l1_minimization(A,b,n,opts)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 09/15/2017


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original motivation to find:

%               min_f {  mu ||Af - b||_1 + ||D^k f||_1 }

% where D^k is kth order finite difference
% The problem is modified using variable splitting
% and this algorithm solves: 

%      min_{f,w} { beta/2 ||mu (Af - b) - w||_2^2 + ||w||_1
%                   + beta/2 ||Df - v||_2^2 + ||v||_1
%               - (delta , Af - b -w ) - (sigma, Df-v) }

% delta and sigma are Lagrange multiplier
% the equations (Af-b) and (Df - v) are concatonated (stored into Af - b)
% the variables w and v are alos concatenated (stored into w)
% therefore, the variables delta and sigma are also concatenated (into
% delta)
% After concatonating the equations, we solve 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                beta/2 ||(Gf-d) - w ||_2^2 + ||w||_1 - (delta, (Gf-d)-w)
% constants:                  lam1             lam2          lam3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where G^T = (A^T , D^T), d^T = (b^T,zeros) , w is splitting variable 

% Algorithm uses alternating direction minimization over f and w.



% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   b: data values in vector form
%   p,q,r: signal dimensions
%   opts: structer containing opts, see function check_HOTV3D_opts.m for these


% Outputs:
%   U: reconstructed signal
%   out: output numerics

p = n(1); q = n(2); r = n(3);



% get and check opts
opts = check_HOTV_opts(opts);


% mark important constants
tol_inn = opts.tol_inn;
tol_out = opts.tol_out;
k = opts.order;
n = p*q*r;



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

[D,Dt] = get_D_Dt(k,p,q,r,opts);

A = @(u,mode)l1_l1_operator(A,D,Dt,u,mode,numel(b),(opts.mu));
tmp = D(zeros(p*q*r,1));
b = opts.mu*[b;zeros(numel(tmp),1)];
clear tmp;


% sanity check
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg
    error('A and A* do not appear consistent');
end
clear flg;

% calculate A'*b
Atb = A(b,2);


% initialize everything else 
[U,beta,betaf,delta,gL,out] ...
    = get_l1solver(p,q,r,Atb,A(Atb,1),scl,opts,b);    % U: p*q

nrmb = norm(b);
Upout = U;
Aub = A(U,1)-b;



% first shrinkage step
W = max(abs(Aub) - 1/beta, 0).*sign(Aub);

lam2 = sum(sum(sum(abs(W))));

% gA and gD are the gradients of ||Au-b||^2 and ||Du-w||^2, respectively
% i.e. g = A'(Au-b), gD = D'(Du-w)
[lam1,lam3,f,Aub,gA] = get_grad(U,W,lam2,beta,A,b,delta);


% compute gradient
g = beta*gA - gL;


out.f = [out.f; f]; 
out.lam1 = [out.lam1; lam1]; 
out.lam2 = [out.lam2; lam2]; 
out.lam3 = [out.lam3; lam3];

for ii = 1:opts.outer_iter
    if opts.disp
            fprintf('    Beginning outer iteration #%d\n',ii);
            fprintf('   beta = %d , order = %g\n',beta,k);
            fprintf('iter    ||Au-b-w||_2  ||w||^1      f\n');
    end
        
    %initialize the constants
    gam = opts.gam; Q = 1; fp = f;
    
    
    for jj = 1:opts.inner_iter
        % compute step length, tau
        %if jj~=1
        if jj==0
            % BB-like step length
            dgA = gA - gAp;   
            %dgD = gD - gDp;                    
            ss = uup'*uup;                      
            sy = uup'*(beta*dgA);       
            tau = abs(ss/max(sy,eps));      
        else
            % do Steepest Descent at the 1st ieration
            Ag = A(g,1);
            tau = abs((g'*g)/(beta*(Ag')*Ag));
        end

        % keep previous values for backtracking & computing next tau
        Up = U; gAp = gA; %Aup = Au; 
        Aubp = Aub; %DtsAtdp =  DtsAtd;
        fp = f;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ONE-STEP GRADIENT DESCENT %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = U - tau*g;
        % projected gradient method for inequality constraints
        if opts.nonneg
            U = max(real(U),0);
        elseif opts.isreal
            U = real(U);
        end
        if opts.max_c
            U = min(U,max_v);
        end
        
        % update gradient
        [lam1,lam3,f,Aub,gA] = get_grad(U,W,lam2,beta,A,b,delta);

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
                    dAu = Au - Aup;
                    flg = false;
                end
                % shrink alpha
                alpha = alpha*opts.gamma;
                % U is moved back toward Up, in particular: 
                % U = alpha*U +(1-alpha)Up;
                % all other values are updated accordingly
                [U,lam1,lam2,f,Au,gA] = back_up(p,q,r,lam3,...
                    alpha,beta,Up,du,gAp,dgA,Aup,dAu,W,b,delta)
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
                % shrinkage
                Aubar = Aub - delta/beta;
                W = max(abs(Aubar) - 1/beta, 0).*sign(Aubar);
                
                [lam1,lam3,f,Aub,gA] = get_grad(U,W,lam2,beta,A,b,delta);
                alpha = 0; % remark the failure of back tracking
                break;
            end
            
        end

        % if back tracking is successful, then recompute
        if alpha ~= 0
            Aubar = Aub - delta/beta;
            W = max(abs(Aubar) - 1/beta, 0).*sign(Aubar);
            % update parameters related to W
            [lam1,lam2,lam3,f,gA] = update_W(A,beta,W,Aub,delta);
        end

        % update reference value
        Qp = Q; Q = gam*Qp + 1; fp = (gam*Qp*fp + f)/Q;
        uup = U - Up; uup = uup(:);           % uup: pqr
        rel_chg_inn = norm(uup)/norm(Up(:));
        
        
        
        out.f = [out.f; f]; 
        out.lam1 = [out.lam1; lam1]; 
        out.lam2 = [out.lam2; lam2]; 
        out.lam3 = [out.lam3; lam3];
        out.tau = [out.tau; tau]; 
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        out.cnt = [out.cnt;cnt];
       

        
        
        
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            prnt_format = '%3.0f %10.5g %12.5g %13.5g\n';
            fprintf(prnt_format, jj,lam1,lam2,f);
        end


        % recompute gradient
        g = beta*gA - gL;
        
        % move to next outer iteration and update multipliers if relative
        % change is less than tolerance
        if (rel_chg_inn < tol_inn), break; end;
        
        
    end
    % end of inner loop
    
    
    rel_chg_out = norm(U(:)-Upout(:))/norm(Upout(:));
    out.rel_chg_out = [out.rel_chg_out; rel_chg_out];
    Upout = U;

    % stop if already reached optimal solution
    if rel_chg_out < tol_out
        break;
    end

    % update multipliers
    [delta,lam3] = update_mlp(beta,W,Aub,delta);
    


    % update penality parameters for continuation scheme
    %beta0 = beta;
    beta = min(betaf, beta*opts.rate_ctn);
    

    % update function value, gradient, and relavent constant
    f = beta/2*lam1 + lam2 - lam3;
    %gL = -(beta0/beta)*g;     % DtsAtd should be divided by new beta  
    gL =  A(delta,2);
    % gradient
    g = beta*gA - gL;

end

out.total_iter = numel(out.f)-1;
%out.final_error = norm(A(U(:),1)-b)/nrmb;
out.final_error = sum(abs(A(U(:),1)-b))/sum(abs(b));
out.final_wl1 = lam1(end);
out.final_Du_w = lam2(end);
%out.rel_error = sqrt(out.lam3)/nrmb;


%final_disp(out,opts);
            
% rescale U
U = reshape(U/scl,p,q,r);



function [lam1,lam3,f,Aub,gA] = get_grad(U,W,lam2,beta,A,b,delta)

Aub = A(U,1) - b;
Aubw = Aub-W;
% gA = A'(Au-b)
gA = A(Aubw,2);

lam1 = (Aubw)'*(Aubw);

%lam3
lam3 = delta'*(Aub-W);

% f
f = beta/2*lam1 + lam2 - lam3;





         
function [lam1,lam2,lam3,f,gA] = update_W(A,beta,W,Aub,delta)


% update parameters because W was updated
lam2 = sum(sum(sum(abs(W))));
Aubw = Aub - W;

gA = A(Aubw,2);
lam1 = (Aubw)'*(Aubw);
lam3 = delta'*Aubw;
f = beta/2*lam1 + lam2 - lam3;


function [delta,lam3] = update_mlp(beta,W,Aub,delta)



Aubw = Aub-W;
delta = delta - beta*Aubw;

%tmpf = f + lam4 + lam5;
lam3 = delta'*Aubw;

function [U,lam1,lam3,f,Au,gA] = back_up(p,q,r,lam2,...
    alpha,beta,Up,du,gAp,dgA,Aup,dAu,W,b,delta)

gA = gAp + alpha*dgA;
U = Up + alpha*reshape(du,p,q,r);
Au = Aup + alpha*dAu;
Aubw = Au-b-W;

lam1 = (Aubw)'*(Aubw);
lam3 = delta'*Aubw;
f = beta/2*lam1 + lam2 - lam3;

function y = l1_l1_operator(A,D,Dt,u,mode,d1,mu)

% this function concatonates the two terms
switch mode
    case 1
         y = mu*A(u,1);
         y = [y;col(D(u))];
    case 2
         y = mu*A(u(1:d1),2);
         y = y + Dt(u(d1+1:end));
end


