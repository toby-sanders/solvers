function [U,delta,gL,out] = get_l1solver(p,q,r,A,scl,opts,b)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 12/18/2017

% initialize D^T sigma + A^T delta
gL = zeros(p*q*r,1);



% declare out variables
out.rel_chg_inn = [];  out.rel_chg_out = []; out.rel_lam2 = [];
out.f = [];        % record values of augmented Lagrangian fnc
out.cnt = [];      % record # of back tracking
out.lam1 = []; out.lam2 = []; out.lam3 = []; out.lam4 = []; out.lam5 = [];
out.tau = []; out.alpha = []; out.C = []; out.mu = [];
out.DU = [];out.rel_error = [];

if opts.store_soln
    if r~=1
        warning('Solution is not being stored since r~=1')
        opts.store_soln = false;
    else
        out.Uall = zeros(p,q,opts.inner_iter*opts.outer_iter);
    end
end

% initialize U
[mm,nn,rr] = size(opts.init);

% iterations used to generate an initial starting point
if ~isfield(opts,'init_iter')
    opts.init_iter = 10;
end

if max([mm,nn,rr]) == 1
    % basic gradient descent for initial solution
    [U,~] = basic_GD_local(A,b,p,q,r,opts.init_iter);
else
    if mm ~= p || nn ~= q || rr ~= r
        fprintf('Input initial guess has incompatible size! Switch to the default initial guess. \n');
        % basic gradient decent for initial solution
        [U,~] = basic_GD_local(A,b,p,q,r,opts.init_iter);
    else
        fprintf('user supplied intial solution\n');
        U = opts.init*scl;
    end
end
U = U(:);


% initialize multiplers                     
delta = zeros(length(b),1);



function [U,out] = basic_GD_local(A,b,p,q,r,iter)

% algorithm for basic gradient decent for LS

% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end


U = zeros(p*q*r,1);
out.alpha = zeros(iter,1);
out.rel_error = zeros(iter,1);
nrmb = b'*b;
for i = 1:iter
   Aub = A(U,1)-b;
   g = A(Aub,2); 
   Ag = A(g,1);
   alpha = Ag'*Aub/max(Ag'*Ag,eps);  % steepest decent value
   U = U - alpha*g;
   out.alpha(i) = alpha;
   out.rel_error(i) = Aub'*Aub/nrmb;
end
U = reshape(U,p,q,r);