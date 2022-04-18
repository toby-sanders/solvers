function [x,out] = myLSineq(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx - d >= 0
% using a standard lagrange multiplier method


if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'tol'), opts.tol = 1e-5; end

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
if ~isa(C,'function_handle'), C = @(u,mode) f_handleA(C,u,mode); end

beta = 0;% beta = 32/max(abs(b));
gam = .1; % step size for LM update

Atb = A(b,2);
x = zeros(size(Atb));
lambda = zeros(size(d));
out.rel_chg = [];
for ii = 1:opts.iter
    g1 = A(A(x,1),2) - Atb; % gradient over quadratic term
    g2 = C(lambda,2);   % gradient on LM term
    g3 = beta*C(C(x,1)-d,2); % gradient over quadratice term (not used here)
    g = g1+g2+g3;
    Ag = A(g,1);
    Cg = C(g,1);
    % could try a BB-step
    if mod(ii-1,10)==0
        tau = (g'*g1 + lambda'*C(g,1))/(Ag'*Ag + beta*(Cg')*Cg); % step length
    else
        st = x-xp;
        yp = g-gp;
        tau =  (st'*st)/(st'*yp);
    end
    
    xp = x;
    gp = g;
    x = x - tau*g; % descent
    if mod(ii,10)==0 % Lagrange multiplier update 
        lambda = lambda + gam*(C(x,1)-d); 
        
        % multipliers for positive inequality contraint are non-positive
        lambda = min(lambda,0); 
    end
    out.rel_chg = [out.rel_chg;myrel(x,xp)]; 
    if out.rel_chg(end)<opts.tol, break; end % check convergence
end
out.lambda = lambda;