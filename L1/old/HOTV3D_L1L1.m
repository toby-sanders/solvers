function [U, out] = HOTV3D_L1L1(A,b,n,opts)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 12/18/2017


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original motivation to find:

%               min_f {  mu ||Af - b||_1 + ||D^k f||_1 }

% where D^k is kth order finite difference
% The problem is modified using variable splitting and this algorithm 
% works with the following augmented Lagrangian function: 
%
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
% where G^T = (mu A^T , D^T), d^T = mu (b^T,zeros) , w is splitting variable 

% Algorithm uses alternating direction minimization over f and w.



% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   b: data values in vector form
%   p,q,r: signal dimensions
%   opts: structure containing opts, see function check_HOTV3D_opts.m for these


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
mu = opts.mu/4; % works well for this algorithm
muf = opts.mu;

% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end

% check scaling A
if opts.scale_A
    [A,b] = ScaleA(n,A,b);
end

% scale beta
[~,scl] = Scaleb(b);
beta = opts.beta*scl;

% check for maximum constraint value
if opts.max_c
    max_v = opts.max_v;
end

[D,Dt] = get_D_Dt(k,p,q,r,opts);
B = A;
numb = numel(b);

A = @(u,mode)l1_l1_operator(A,D,Dt,u,mode,numb,mu);
tmp = D(zeros(p*q*r,1));
b = mu*[b;zeros(numel(tmp),1)];
clear tmp;


% sanity check
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg
    error('A and A* do not appear consistent');
end
clear flg;

% initialize everything else 
[U,delta,gL,out] = get_l1solver(p,q,r,A,1,opts,b);    % U: p*q


nrmb = sum(abs(b));
Upout = U;
Aub = A(U,1)-b;



% first shrinkage step
W = max(abs(Aub) - 1/beta, 0).*sign(Aub);

lam2 = sum(sum(sum(abs(W))));

% gradient and obj functional
[lam1,lam3,f,Aub,Au,gA] = get_grad(U,W,lam2,beta,A,b,delta);


% compute gradient
g = beta*gA - gL;


out.f = [out.f; f]; 
out.lam1 = [out.lam1; lam1]; 
out.lam2 = [out.lam2; lam2]; 
out.lam3 = [out.lam3; lam3];
out.rel_error = [out.rel_error; sum(abs(Aub))];
out.rdelta = [];
for ii = 1:opts.outer_iter
    if opts.disp
            fprintf('    Beginning outer iteration #%d\n',ii);
            fprintf('   beta = %d , order = %g\n',beta,k);
            fprintf('iter    ||Au-b-w||_2  ||w||^1   ideal obj     true obj\n');
    end
        
    for jj = 1:opts.inner_iter
        % compute step length, tau
            if jj~=1
                % BB-like step length
                dgA = gA - gAp;                   
                ss = uup'*uup;                      
                sy = uup'*(beta*dgA);       
                tau = abs(ss/max(sy,eps));      
            else
                % optimize tau at the 1st ieration
                Ag = A(g,1);
                tau = ((g'*g)/(beta*(Ag')*Ag));
            end

            % keep previous values for backtracking & computing next tau
            Up = U; gAp = gA; fp = f;
            
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
            [lam1,lam3,f,Aub,Au,gA] = get_grad(U,W,lam2,beta,A,b,delta);
            %g = beta*gA - gL;
            %uup = U - Up; uup = uup(:); 
            cnt = 0;
            % if obj function increases, just optimize tau
            if fp<f
                U = Up;
                cnt = 1; % mark that tau is being optimized
                %fprintf('1 step steepest decent\n');
                for kk = 1
                   Ag = A(g,1);
                   tau = ((g'*g)/(beta*(Ag')*Ag));
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
            
                   %[lam1,lam3,f,Aub,Au,gA] = get_grad(U,W,lam2,beta,A,b,delta);
                   Aub = A(U,1)-b;
                   gA = A(Aub-W,2);
                   g = beta*gA-gL;
                end               
            end
            
                
        % shrinkage
        W = Aub - delta/beta;
        W = max(abs(W) - 1/beta, 0).*sign(W);
        % update parameters related to W
        [lam1,lam2,lam3,f,gA] = update_W(A,beta,W,Aub,delta);


        % update reference values
        uup = U - Up; uup = uup(:);
        rel_chg_inn = norm(uup)/norm(Up(:));
        
        
        
        out.f = [out.f; f]; 
        out.lam1 = [out.lam1; lam1]; 
        out.lam2 = [out.lam2; lam2]; 
        out.lam3 = [out.lam3; lam3];
        out.tau = [out.tau; tau]; 
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        out.rel_error = [out.rel_error;sum(abs(Aub))/nrmb];
        out.cnt = [out.cnt;cnt];
       

        
        
        
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            prnt_format = '%3.0f %10.5g %12.5g %13.5g %13.5g\n';
            fprintf(prnt_format, jj,lam1,lam2,out.rel_error(end),f);
        end


        % recompute gradient
        g = beta*gA - gL;
        
        % move to next outer iteration and update multipliers if relative
        % change is less than tolerance
        if (rel_chg_inn < tol_inn), break; end
        
        
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
    deltap = delta;
    [delta,lam3] = update_mlp(beta,W,Aub,delta);
    out.rdelta = [out.rdelta;norm(delta-deltap)/norm(deltap)];


    % update penality parameters for continuation scheme
    mup = mu;
    mu = min(muf, mu*opts.rate_ctn);   
    A = @(u,mode)l1_l1_operator(B,D,Dt,u,mode,numb,mu);
    b = b*mu/mup;
    nrmb = sum(abs(b));
    
   
    % update function value, gradient, and relavent constant
    [lam1,lam3,f,Aub,Au,gA] = get_grad(U,W,lam2,beta,A,b,delta);
    
    % gradient for lam3
    gL =  A(delta,2);
    % gradient
    g = beta*gA - gL;

end

out.total_iter = numel(out.f)-1;
%out.final_error = norm(A(U(:),1)-b)/nrmb;
out.final_error = f;%sum(abs(A(U(:),1)-b))/nrmb;
out.final_wl1 = lam2(end);
out.final_Du_w = lam1(end);
%out.rel_error = sqrt(out.lam3)/nrmb;


final_disp_l1_l1(out,opts);
            
% rescale U
U = reshape(U,p,q,r);


function [lam1,lam3,f,Aub,Au,gA] = get_grad(U,W,lam2,beta,A,b,delta)

Au = A(U,1);
Aub = Au - b;
Aubw = Aub-W;
% gA = A'(Au-b)
gA = A(Aubw,2);

lam1 = (Aubw)'*(Aubw);

%lam3
lam3 = delta'*(Aubw);

% f
f = beta/2*lam1 + lam2 - lam3;





         
function [lam1,lam2,lam3,f,gA] = update_W(A,beta,W,Aub,delta)


% update parameters because W was updated
lam2 = sum(sum(sum(abs(W))));
Aubw = Aub - W;

gA = A(Aubw,2);
lam1 = (Aubw)'*Aubw;
lam3 = delta'*Aubw;
f = beta/2*lam1 + lam2 - lam3;


function [delta,lam3] = update_mlp(beta,W,Aub,delta)



Aubw = Aub-W;
delta = delta - beta*Aubw;

%tmpf = f + lam4 + lam5;
lam3 = delta'*Aubw;


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


