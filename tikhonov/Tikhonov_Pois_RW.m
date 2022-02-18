function [x,out] = Tikhonov_Pois_RW(A,b,n,opts)

% Poisson model
% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||R(Ax - b)||_2^2 + ||Dx||^2
% where D^k is kth order finite difference, and R is a diagonal matrix with
% weights estimated by R_{ii} = (sqrt((Au)_i) + epsilon)^-1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D

% *************** VERY IMPORTANT (AND USEFUL)  ******************
% The important parameter mu is automatically apdapted using the 
% "discrepancy principle" (the noise variance in the data should match the 
% means), i.e. E[ ||Au-b||^2 ] =  sum(Au) 
% HOWEVER, THIS WILL ONLY WORK PROPERLY IF THE DATA IS TRUELY POISSON AND 
% HAS NOT BEEN RESCALED IN ANY MANNER.
%  **************************************************************

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 06/25/2018

% get initial solution
tmp = opts;tmp.iter = 5;
tmp.mu = 500;tmp.disp = false;
x = Tikhonov_CG(A,b,n,tmp);x = x(:);
clear tmp;

if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3); n = p*q*r;
opts = check_tik_opts(opts);

% mark important constants
mu = opts.mu; iter = opts.iter;
tol = opts.tol; k = opts.order;
nrmb = b'*b;

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end

% check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

% initialize out
out = init_out(iter,mu);
[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,A(b,2));
[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg, error('D and Dt do not appear consistent'); end
clear flg;
Ax = A(x,1);    % Ax
dd = D(x);  % Dx

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is for Poisson noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepmu = .2;
flg_inc = 1;flg_dec = 1;
max_mu = 1e5;min_mu = 1e-7;
epsilon = .1;
if sum(b<0), warning('data vector b should be nonnegative');end
% create scaled b for weights
btmp = b/max(b);
R = 1./(sqrt(btmp)+epsilon); % proper weights assuming poisson noise
R = R/max(R);   % rescale them
% save the original operator and original b for later
AO = A;  borig = b;  
% weighted operator and weighted b
A = @(u,mode)my_local_oper(A,R,u,mode);
Rb = b.*R;
Ax = R.*Ax;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:iter    
    % gradient
    gA = A(Ax-Rb,2);
    g = mu*gA+Dt(dd); 
 
    % step length, tau
    if i==1
        Ag = A(g,1); Dg = D(g);
        tau = g'*g/(mu*(Ag'*Ag) + col(Dg)'*col(Dg)); % opt. step length
    else
        tau = (xxp'*xxp)/(xxp'*(g-gp)); % BB-step
    end
    xp = x; % save previous solution
    out.tau(i) = tau; % save tau and norm of gradient
    out.g(i) = norm(g);
    
    % decend, project, and recompute forward operations
    x = x - tau*g;
    x = max(real(x),0);
    Ax = A(x,1); dd = D(x);
    
    out.rel_error(i) = norm(Ax-b,2)/nrmb;
    out.nrm_reg(i) = norm(dd);
    out.obj_func(i) = mu*out.rel_error(i) + out.nrm_reg(i);   
    if opts.disp
        fprintf('iter = %i, E(x) = %g, ||Dx|| = %g, obj_func = %g, mu = %g\n',...
            i,out.rel_error(i),out.nrm_reg(i),out.obj_func(i),mu);
    end
       
    % check for convergence
    if norm(x-xp)/norm(xp)<tol
        out.rel_error = out.rel_error(1:i);
        out.nrm_reg = out.nrm_reg(1:i);
        out.tau = out.tau(1:i);
        out.obj_func = out.obj_func(1:i);
        fprintf('convergence tolerance achieved\n\n');
        break;
    end
    
    % save previous values for BB step length
    gp = g; xxp = x-xp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is for Poisson noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if mod(i,5)==0           
        % update weights for Poisson noise
        Rp = R;
        R = AO(x(:),1);    % new estimate for b
        %[btmp,~] = Scaleb(btmpo);
        R = R/max(R);
        R = 1./(sqrt(R)+epsilon); % weights
        R = R/max(R);   % rescale them
        out.R = [out.R;myrel(R,Rp)];
        % new weighted operator and weighted b and A'*b
        A = @(u,mode)my_local_oper(AO,R,u,mode);
        Rb = b.*R;
    end
    % update mu periodically
     if i>50 && mod(i,10)==0
        [mu,flg_inc,flg_dec,out] =...
            get_mu(AO(x,1),b,mu,stepmu,max_mu,min_mu,flg_dec,flg_inc,out);
     end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% output final solution
x = reshape(x,p,q,r);

if opts.disp 
    figure(17);hold off;
    subplot(4,1,1);
    plot(out.rel_error);xlabel('iteration');ylabel('relative data fit error');
    subplot(4,1,2);
    plot(out.obj_func);xlabel('iteration');ylabel('obj func');
    subplot(4,1,3);
    %plot(out.tau);xlabel('iteration');ylabel('step length');
    plot(out.nrm_reg);xlabel('iteration');ylabel('||Dx||');
    subplot(4,1,4);
    plot(out.g);xlabel('iteration');ylabel('gradient norm');
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

% check if condition is approximately satisfied
if abs(1 - r1/r2)>1e-2
   r1 = 1-r1/r2; % compute signed difference to determine change needed
   out.r1 = [out.r1;r1];
   if r1 < 0  % increase mu
       if flg_inc == 0
           stepmu = stepmu*.5;
       end
       flg_inc = 1;
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
out.mu = [out.mu;mu];


function out = init_out(iter,mu)
% initialize output variables
out.rel_error = zeros(iter,1);
out.nrm_reg = zeros(iter,1);
out.tau = zeros(iter,1);
out.g = zeros(iter,1);
out.obj_func = zeros(iter,1);
out.mu = mu;
out.r1 = [];
out.taus = [];
out.R = [];
