function [x,out] = Tikhonov_SD(A,b,n,opts)

% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest decent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% tol - convergence tolerance for CG
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/1/2018

% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = p*q*r;

% mark important constants
mu = opts.mu;
iter = opts.iter;
tol = opts.tol;

% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end

%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

% check scaling A
% if opts.scale_A, 
fprintf('A is always scaled in Nesta version\n');
[A,b] = ScaleA(p*q*r,A,b); 
% end

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
if ~flg, error('D and Dt do not appear consistent'); end
clear flg;

nrmb = norm(b);
e = A(x,1)-b;  % Ax - b
dd = D(x);   % Dx
Atb = A(b,2);
y = x;
tic;
fprintf('Running Tikhonov solver...\n');
for i = 1:iter
      
     g = mu*A(e,2)+Dt(dd); % gradient
    % g = mu*A(A(x,1),2) - Atb + Dt(D(x));
      tau = 1/mu;
    % step length, tau
%     if i==1 || iter-i < 3
%        Ag = A(g,1); Dg = D(g);
%       tau = g'*g/(mu*(Ag'*Ag) + col(Dg)'*col(Dg)); % steepest decent
%      else
%         tau = (xxp'*xxp)/(xxp'*(g-gp)) % BB-step
%      end
    % gradient decent
    xp = x;
    x = y - tau*g;
    y = x + (i-1)/(i+2)*(x-xp);
    
%     figure(997);
%     subplot(2,2,3);hold off;
%     plot(x);hold on;
%     plot(y);hold off;
    % projected gradient method for inequality constraints
    if opts.nonneg, x = max(real(x),0);
    elseif opts.isreal, x = real(x); end
    if opts.max_c, x = min(x,max_v); end
    
    e = A(y,1)-b;  % Ax - b
    % e = A(y,1)-b;
    dd = D(y);   % Dx
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
        fprintf('convergence tolerance achieved\n');
        break;
    end
    % save previous values for BB step length
    gp = g;
    xxp = x-xp;
    
end
fprintf('total iterations = %i\n\n',i);
out.total_time = toc;
out.iters = i;
out.g = g;
% output final solution
x = reshape(x,p,q,r);