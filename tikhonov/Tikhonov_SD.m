function [x,out] = Tikhonov_SD(A,b,n,opts)

% this version of the tikhonov solver uses gradient descent with variable
% step lengths in each iteration, which are determined using the BB method


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
mu = opts.mu;


% unify implementation of A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end
%check that A* is true adjoint of A
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

% initialize out and x
out.rel_chg = zeros(opts.iter,1);
x = zeros(p*q*r,1);

[D,Dt] = get_D_Dt(opts.order,p,q,r,opts);
[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg
    error('D and Dt do not appear consistent');
end
clear flg;

tic;
for i = 1:opts.iter
      
    e = A(x,1)-b;  % Ax - b
    dd = D(x);   % Dx
    g = mu*A(e,2)+Dt(dd); % gradient
    % determine step length, tau
    if i==1 || opts.iter-i < 3
        Ag = A(g,1); Dg = D(g);
        tau = g'*g/(mu*(Ag'*Ag) + col(Dg)'*col(Dg)); % optimal single step
    else
        tau = (xxp'*xxp)/(xxp'*(g-gp)); % BB-step
    end
    xp = x; % save previous solution
    x = x - tau*g; % gradient descent
    
    % projected gradient method for inequality constraints
    if opts.nonneg, x = max(real(x),0);
    elseif opts.isreal, x = real(x); end

    % check for convergence
    out.rel_chg(i) = norm(x-xp)/norm(xp);
    if out.rel_chg(i) < opts.tol
        out.rel_chg = out.rel_chg(1:i);
        break;
    end
    % save previous values for BB step length
    gp = g;
    xxp = x-xp;
    
end
out.total_time = toc;
out.iters = i;
out.g = g;
% output final solution
x = reshape(x,p,q,r);