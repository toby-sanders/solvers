function [x,out] = Tikhonov_Nesta(A,b,n,opts)

% this version uses standard gradient descent with acceleration using the
% Nesterov method/heavy ball approach


% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% using a simple steepest descent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% tol - convergence tolerance
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/1/2018

% set image dimensions
opts = check_tik_opts(opts);
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = p*q*r;
mu = opts.mu;

% unify implementation of A
if ~opts.A2
    if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode);end
    %check that A* is true adjoint of A
    [flg,rel_diff] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
    if ~flg
        error('A and A* operator mismatch.\n Rel. difference in test was %g',rel_diff); 
    end
    clear flg rel_diff;
    AtA =  @(x)A(A(x,1),2);
    b = A(b,2);
else % case the operator AtA is input
    % in this case, the data vector, b, should actually be A^T*b
    if numel(b)~=p*q*r
        error('if A^T*A is input, data vector b should be A^T*b');
    end
    AtA = A;
end

% initialize out and x
out.rel_chg = zeros(opts.iter,1);
out.objF = [];
if ~isempty(opts.init)
    x = opts.init(:);
else
    x = zeros(p*q*r,1);
end

% get the step length for the objective function
[tau,out.tauStuff] = getStepLength_SPD(AtA,n);
if isfield(opts,'regV')
    if ~isempty(opts.regV)
        L = abs(opts.regV).^2;
    else
        L = my_Fourier_filters(opts.order,opts.levels,p,q,r);
    end
else
    L = my_Fourier_filters(opts.order,opts.levels,p,q,r);
end
L = mu/tau + max(L(:)); % lipchitz constant for combined operators
tau = (1/L);

[D,Dt] = get_D_Dt(opts.order,p,q,r,opts);
[flg,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg, error('D and Dt operator mismatch'); end
clear flg;

xp = x;
tic;
for i = 1:opts.iter
    
    y = x + (i-1)/(i+2)*(x-xp); % new accelerated vector
    xp = x;
    g = mu*(AtA(y) - b) + Dt(D(y));
    x = y - tau*g; % gradient descent from accelerated vector, y    
    
    if opts.nonneg, x = max(real(x),0);
    elseif opts.isreal, x = real(x); end   
    
    % check for convergence
    out.rel_chg(i) = norm(x-xp)/norm(xp);
    if out.rel_chg(i) < opts.tol
        out.rel_chg = out.rel_chg(1:i);
        break;
    end
    if opts.disp
        Axb = A(x,1)-b;Axb = mu*Axb(:)'*Axb(:);
        Dx = D(x);Dx = Dx(:)'*Dx(:);
        objF = Axb + Dx;
        out.objF = [out.objF;objF,Axb,Dx];
        figure(6);
        tiledlayout(2,2);
        nexttile;imagesc(reshape(x,p,q,r));colorbar;
        title(i);
        colorbar;
        nexttile;
        semilogy(out.rel_chg);
        ylabel('relative change');
        xlabel('iteration');title('convergence');
        nexttile;
        semilogy(out.objF(:,1));
        ylabel('objective function');
        xlabel('iteration');title('convergence');
        nexttile;hold off;
        semilogy(out.objF(:,2));hold on;
        semilogy(out.objF(:,3));hold off;
        legend('||Axb||','||Dx||');hold off;
        xlabel('iteration');title('convergence');
    end
    
end
out.total_time = toc;
out.iters = i;
out.g = g;
% output final solution
x = reshape(x,p,q,r);