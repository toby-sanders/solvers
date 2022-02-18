function [x,out] = Tikhonov(A,b,n,C,opts)

% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest decent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 10/08/2017

% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = p*q*r;
opts = check_tik_opts(opts);

% mark important constants
mu = opts.mu;
iter = opts.iter;
tol = opts.tol;
k = opts.order;

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end

%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

% check scaling A
if opts.scale_A, [A,b] = ScaleA(p*q*r,A,b); end

% initialize out and x
out.rel_error = zeros(iter,1);
out.nrm_reg = zeros(iter,1);
out.obj_func = zeros(iter,1);
out.rel_chg = zeros(iter,1);
x = zeros(p*q*r,1);


[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,A(b,2));
%rescale for the l2 norm
% if k~=0
%     D = @(x)D(x)*2^(k-1)/sqrt(nchoosek(2*(k-1),k - 1));
%     Dt = @(x)Dt(x)*2^(k-1)/sqrt(nchoosek(2*(k-1),k - 1));
% end
[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg
    error('D and Dt do not appear consistent');
end
clear flg;

nrmb = norm(b);
e = A(x,1)-b;  % Ax - b
dd = D(x);   % Dx
sigma = 0;
tic;
fprintf('Running Tikhonov solver...');
for i = 1:iter
    
    g = mu*(e'*e - C)*A(e,2) + Dt(dd) - sigma*A(e,2); % gradient
    % step length, tau
    if i==1 %|| flg == 1% || i> iter - 3
        Ag = A(g,1); Dg = D(g);
        tau = g'*g/(mu*(Ag'*Ag) + col(Dg)'*col(Dg)); % steepest decent
    else
        tau = (xxp'*xxp)/(xxp'*(g-gp)); % BB-step
    end
    xp = x; % save previous solution
    % gradient decent
    x = x - tau*g;
    
    % projected gradient method for inequality constraints
    if opts.nonneg, x = max(real(x),0);
    elseif opts.isreal, x = real(x); end
    if opts.max_c, x = min(x,max_v); end
    
    e = A(x,1)-b;  % Ax - b
    dd = D(x);   % Dx
    out.rel_error(i) = norm(e)/nrmb;
    out.nrm_reg(i) = norm(dd);
    out.obj_func(i) = mu*out.rel_error(i) + out.nrm_reg(i);
    out.rel_chg(i) = norm(x-xp)/norm(xp);
    
    % check for convergence
    if out.rel_chg(i) < tol
        out.rel_error = out.rel_error(1:i);
        out.nrm_reg = out.nrm_reg(1:i);
        out.obj_func = out.obj_func(1:i);
        out.rel_chg = out.rel_chg(1:i);
        fprintf('convergence tolerance achieved\n\n');
        break;
    end
    % save previous values for BB step length
    gp = g;
    xxp = x-xp;
    flg = 0;
    if mod(i,20)==0, sigma = sigma - mu*(e'*e - C); flg = 1;end
end
out.total_time = toc;
fprintf('completed in %g seconds\n',out.total_time);
fprintf('iters = %i, ||Ax-b||/||b|| = %g\n',i,out.rel_error(i));
out.g = g;
% output final solution
x = reshape(x,p,q,r);

% if opts.disp 
%     figure(18);hold off;
%     subplot(3,1,1);
%     plot(out.rel_error);xlabel('iteration');ylabel('L2 error');
%     subplot(3,1,2);
%     plot(out.nrm_reg);xlabel('iteration');ylabel('reg norm');
%     subplot(3,1,3);
%     plot(out.obj_func);xlabel('iteration');ylabel('obj func');
% end
