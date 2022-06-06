function [x,out] = myLSineq(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx - d >= 0
% using a standard lagrange multiplier method


if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'tol'), opts.tol = 1e-5; end

% if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
% if ~isa(C,'function_handle'), C = @(u,mode) f_handleA(C,u,mode); end

beta = 0;% beta = 32/max(abs(b));
gam = .01; % step size for LM update

Atb = A'*b;% A(b,2);
x = zeros(size(Atb));
lambda = zeros(size(d));
out.rel_chg = [];
g2 = 0;
AtA = A'*A;
for ii = 1:opts.iter
    g1 = AtA*x - Atb;% A'*(A*x) - Atb; % gradient over quadratic term   
    g = g1+g2;
    
    % get step length, tau
    if ii~=0% mod(ii-1,10)==0
        Ag = A*g;
        Cg = C*g;
        tau = (g'*g1 + lambda'*Cg)/(Ag'*Ag); % step length
    else % bb-step
        st = x-xp;
        yp = g-gp;
        tau =  (st'*st)/(st'*yp);
    end
    
    xp = x;
    gp = g;
    x = x - tau*g; % descent


    % Lagrange multiplier update 
    % multipliers for positive inequality contraint are non-positive
    if mod(ii,10)==0 
        lambda = lambda + gam*(C*x-d); 
        lambda = min(lambda,0); 

        g2 = C'*lambda;   % update gradient on LM term
    end
    out.rel_chg = [out.rel_chg;myrel(x,xp)]; 
    if out.rel_chg(end)<opts.tol, break; end % check convergence
end
out.lambda = lambda;