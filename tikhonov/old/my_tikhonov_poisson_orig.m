function [x,out] = my_tikhonov_poisson(A,b,n,opts)

% Poisson model


% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% min_x    mu*E(x) + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest decent method
% E(x) is the negative log likelihood from poisson model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% options are more or less the same as HOTV3D


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 10/08/2017

% set image dimensions

% get initial solution
tmp = opts;
tmp.iter = 5;
tmp.mu = 500;
tmp.disp = false;
x = my_tikhonov(A,b,n,tmp);
x = x(:);
clear tmp;

if numel(n)<3
    n(end+1:3) = 1;
elseif numel(n)>3
    error('n can have at most 3 dimensions');
end
p = n(1); q = n(2); r = n(3);
n = p*q*r;


opts = check_tik_opts(opts);

% mark important constants
mu = opts.mu;
iter = opts.iter;
tol = opts.tol;
k = opts.order;

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

% check scaling A
opts.scale_A = false;
if opts.scale_A
    [A,b] = ScaleA(p*q*r,A,b);
end

% check scaling b
scl = 1;
opts.scale_b = false;
if opts.scale_b
    [b,scl] = Scaleb(b);
end

% initialize out and x
out.rel_error = zeros(iter,1);
out.nrm_reg = zeros(iter,1);
out.tau = zeros(iter,1);
out.g = zeros(iter,1);
out.obj_func = zeros(iter,1);
out.taus = [];
%x = zeros(p*q*r,1);


[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,A(b,2));
%rescale for the l2 norm
if k~=0
    D = @(x)D(x)*2^(k-1)/sqrt(nchoosek(2*(k-1),k - 1));
    Dt = @(x)Dt(x)*2^(k-1)/sqrt(nchoosek(2*(k-1),k - 1));
end
[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg
    error('D and Dt do not appear consistent');
end
clear flg;

epsilon = 0;%.1;
S = find(b~=0);
for i = 1:iter
    % compute forward operations
    
    
    Ax = A(x,1);
    dd = D(x);
    out.rel_error(i) = sum(Ax-b) + sum(b(S).*log(b(S)./(Ax(S))));
    out.nrm_reg(i) = norm(dd);
    out.obj_func(i) = mu*out.rel_error(i) + out.nrm_reg(i);
    if opts.disp
        fprintf('iter = %i, E(x) = %g, ||Dx|| = %g, obj_func = %g\n',...
            i,out.rel_error(i),out.nrm_reg(i),out.obj_func(i));
%         figure(18);
%         plot(x);
%         pause;
    end
    
    gA = A(1-b./(Ax+epsilon),2);
    gA(isnan(gA))=0;
    gA(isinf(gA))=0;
    g = mu*gA+Dt(dd); % gradient
    % step length, tau
    if i==1
        tau_iterations = 10;
        Ag = A(g,1); Dg = D(g);
        %tau = g'*g/(mu*(Ag'*Ag) + col(Dg)'*col(Dg));
        s1 = 2*col(dd)'*col(Dg);  % 2<Df,Dg>
        s2 = mu*sum(Ag); 
        s3 = 2*Dg(:)'*Dg(:);% 2<Dg,Dg>
        tau = 5e-2;
        tau_iter = zeros(tau_iterations+1,1);
        tau_iter(1) = tau;
        for j = 1:tau_iterations       
            tau = (s1+s2-mu*sum(Ag./(Ax-tau*Ag+eps)))/s3;
            tau_iter(j+1) = tau;
        end
        out.taus = [out.taus, tau_iter];
    else
        tau = (xxp'*xxp)/(xxp'*(g-gp));
    end
    tau = max(tau,1e-10);
    %tau = 5e-3;
    xp = x; % save previous solution
    out.tau(i) = tau; % save tau and norm of gradient
    out.g(i) = norm(g);
    
    % steepest decent
    x = x - tau*g;
    %fprintf('tau = %g, grad = %g\n',tau,norm(g));
    % projected gradient method for inequality constraints
    x = max(real(x),0);

    if opts.max_c
        x = min(x,max_v);
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
    
    % figure(44);plot(x);pause;
    % save previous values for BB step length
    gp = g;
    xxp = x-xp;
    
end

% output final solution
x = reshape(x,p,q,r)/scl;

if opts.disp 
    figure(17);hold off;
    subplot(4,1,1);
    plot(out.rel_error);xlabel('iteration');ylabel('relative data fit error');
    subplot(4,1,2);
    plot(out.obj_func);xlabel('iteration');ylabel('obj func');
    subplot(4,1,3);
    plot(out.tau);xlabel('iteration');ylabel('step length');
    subplot(4,1,4);
    plot(out.g);xlabel('iteration');ylabel('gradient norm');
    
end
