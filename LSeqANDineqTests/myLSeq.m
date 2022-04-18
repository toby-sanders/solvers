function [x,out] = myLSeq(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx = d
% using a standard lagrange multiplier method


if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'tol'), opts.tol = 1e-5; end

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
if ~isa(C,'function_handle'), C = @(u,mode) f_handleA(C,u,mode); end

beta = 1;% beta = 32/max(abs(b));


Atb = A(b,2);
x = zeros(size(Atb));
lambda = zeros(size(d));
out.rel_chg = [];
for ii = 1:opts.iter
    g1 = A(A(x,1),2) - Atb; % gradient over quadratic term
    g2 = C(lambda,2);   % gradient on LM term
    g3 = beta*C(C(x,1)-d,2);
    g = g1+g2+g3;
    Ag = A(g,1);
    Cg = C(g,1);
    % could try a BB-step
    tau = (g'*g1 + lambda'*C(g,1))/(Ag'*Ag + beta*(Cg')*Cg); % step length
    xp = x;
    x = x - tau*g; % decent
    if mod(ii,1)==0 % update LM 
        lambda = lambda + (C(x,1)-d); 
    end
    out.rel_chg = [out.rel_chg;myrel(x,xp)]; 
    if out.rel_chg(end)<opts.tol, break; end % check convergence
end