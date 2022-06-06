function [x,out] = myLSineq(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx - d >= 0
% using a standard lagrange multiplier method


if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'tol'), opts.tol = 1e-5; end
if ~isfield(opts,'LMupdate'), opts.LMupdate = 2; end

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
if ~isa(C,'function_handle'), C = @(u,mode) f_handleA(C,u,mode); end

gam = .00001; % step size for LM update

Atb = A(b,2);
x = zeros(size(Atb));
lambda = zeros(size(d));
out.rel_chg = [];
g2 = 0;
% AtA = A'*A;
for ii = 1:opts.iter
    % gradient over quadratic term   
    g1 = A(A(x,1),2) - Atb;
    g = g1+g2;
    
    % get step length, tau
    if ii==1 
        Ag = A(g,1);
        Cg = C(g,1);
        tau = (g'*g1 + lambda'*Cg)/(Ag'*Ag); % step length
        if isnan(tau)
            tau = 1e-5;
        end
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
    if mod(ii,opts.LMupdate)==0 
        lambda = lambda + gam*(C(x,1)-d);% gam*(C*x-d); 
        lambda = min(lambda,0); 

        g2 = C(lambda,2);% C'*lambda;   % update gradient on LM term
    end
    out.rel_chg = [out.rel_chg;myrel(x,xp)]; 

    if ii>50 % check convergence
        if out.rel_chg(end)<opts.tol, break; end 
    end
end
out.lambda = lambda;



