function [x,out] = my_tikhonov(A,b,n,opts)


% Joe, can you modify this algorithm to solve the Poisson minimization
% problem?

% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% min_x    mu/2*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest decent method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% options are more or less the same as HOTV3D


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 10/08/2017

% set image dimensions
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
% Joe, turn I've turned these scalings off for the Poisson problem
opts.scale_A = false;
opts.scale_b = false;
if opts.scale_A
    [A,b] = ScaleA(p*q*r,A,b);
end

% check scaling b
scl = 1;
if opts.scale_b
    [b,scl] = Scaleb(b);
end

% initialize out and x
out.rel_error = zeros(iter,1);
out.nrm_reg = zeros(iter,1);
x = zeros(p*q*r,1);


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

nrmb = norm(b);
for i = 1:iter
    % compute forward operations
    e = A(x,1)-b;
    dd = D(x);   
    out.rel_error(i) = norm(e)/nrmb;
    out.nrm_reg(i) = norm(dd);
    
    
    g = mu*A(e,2)+Dt(dd); % gradient
    % step length, tau
    % Joe, let's just do steepest decent
    if i~=0
        % steepest decent tau (for l2 problem)
        Ag = A(g,1); Dg = D(g);
        tau = g'*g/(mu*(Ag'*Ag) + col(Dg)'*col(Dg));
    else
        tau = (xxp'*xxp)/(xxp'*(g-gp));
    end
    xp = x; % save previous solution
    
    % steepest decent
    x = x - tau*g;
    
    % projected gradient method for inequality constraints
    if opts.nonneg
        x = max(real(x),0);
    elseif opts.isreal
        x = real(x);
    end
    if opts.max_c
        x = min(x,max_v);
    end
    if opts.disp
        fprintf('iter = %i, ||Ax-b||/||b|| = %g, ||Dx|| = %g\n',...
            i,out.rel_error(i),out.nrm_reg(i));
    end
    
    % check for convergence
    if norm(x-xp)/norm(xp)<tol
        out.rel_error = out.rel_error(1:i);
        out.nrm_reg = out.nrm_reg(1:i);
        fprintf('convergence tolerance achieved\n\n');
        break;
    end
    
    
    % save previous values for BB step length
    gp = g;
    xxp = x-xp;
    
end

% output final solution
x = reshape(x,p,q,r)/scl;

if opts.disp 
    figure(17);hold off;
    subplot(2,1,1);
    plot(out.rel_error);xlabel('iteration');ylabel('relative data fit error');
    subplot(2,1,2);
    plot(out.nrm_reg);xlabel('iteration');ylabel('regularization norm');
end
