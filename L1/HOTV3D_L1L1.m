function [U, out] = HOTV3D_L1L1(A,b,n,opts)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 12/18/2017


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original motivation to find:

%               min_f {  mu ||Af - b||_1 + ||D^k f||_1 }

% where D^k is kth order finite difference
% The problem is modified using variable splitting and this algorithm 
% works with the following augmented Lagrangian function: 
%
%      min_{f,w} { beta/2 ||mu (Af - b) - w||_2^2 + ||w||_1
%                   + beta/2 ||Df - v||_2^2 + ||v||_1
%               - (delta , Af - b -w ) - (sigma, Df-v) }

% delta and sigma are Lagrange multiplier
% the equations (Af-b) and (Df - v) are concatonated (stored into Af - b)
% the variables w and v are also concatenated (stored into w)
% therefore, the variables delta and sigma are also concatenated (into
% delta)
% After concatonating the equations, we solve 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                beta/2 ||(Gf-d) - w ||_2^2 + ||w||_1 - (delta, (Gf-d)-w)
% constants:                  lam1             lam2          lam3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where G^T = (mu A^T , D^T), d^T = mu (b^T,zeros) , w is splitting variable 

% Algorithm uses alternating direction minimization over f and w.



% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   b: data values in vector form
%   p,q,r: signal dimensions
%   opts: structure containing opts, see function check_HOTV3D_opts.m for these


% Outputs:
%   U: reconstructed signal
%   out: output numerics

if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3
    error('n can have at most 3 dimensions');
end
p = n(1); q = n(2); r = n(3);

% get and check opts
opts = check_HOTV_opts(opts);

% mark important constants
tol = opts.tol;
k = opts.order; n = p*q*r;
mu = opts.mu;


% check that A* is true adjoint of A
% check scaling of parameters, maximum constraint value, etc.
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
% if opts.scale_A, [A,b] = ScaleA(n,A,b); end
[D,Dt] = get_D_Dt(k,p,q,r,opts);
numb = numel(b);
A = @(u,mode)l1_l1_operator(A,D,Dt,u,mode,numb,mu);
tmp = D(zeros(p*q*r,1));
b = mu*[b;zeros(numel(tmp),1)];
[~,scl] = Scaleb(b); beta = opts.beta*scl;
clear tmp;
% sanity check
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end; clear flg;

% initialize everything else 
[U,delta,gL,out] = get_l1solver(p,q,r,A,1,opts,b);    % U: p*q
Aub = A(U,1)-b;
W = max(abs(Aub) - 1/beta, 0).*sign(Aub); % first shrinkage step

out.rdelta = [];
while numel(out.rel_chg_inn) <= opts.iter

    % update gradient terms
    Aub = A(U,1)-b;
    gA = A(Aub-W,2); gL =  A(delta,2);
    g = beta*gA - gL;

    % optimize tau at the 1st ieration
    Ag = A(g,1);
    tau = ((g'*g)/(beta*(Ag')*Ag));

    for jj = 1:opts.inner_iter
        % compute BB step length, tau
        if jj~=1
            dgA = gA - gAp;                   
            ss = uup'*uup;                      
            sy = uup'*(beta*dgA);       
            tau = abs(ss/max(sy,eps));      
        end
        
        % gradient descent and projection for inequality constraints
        Up = U; gAp = gA;        
        U = U - tau*g;
        if opts.nonneg, U = max(real(U),0);
        elseif opts.isreal, U = real(U); end
        if opts.max_c, U = min(U,opts.max_v); end

        % shrinkage and update gradient
        Aub = A(U,1)-b;
        W = Aub - delta/beta;
        W = max(abs(W) - 1/beta, 0).*sign(W);
        gA = A(Aub-W,2);
        g = beta*gA - gL;   

        % update reference values
        uup = U - Up; uup = uup(:);
        rel_chg_inn = norm(uup)/norm(Up(:));
                     
        % check inner loop convergence
        out.tau = [out.tau; tau]; 
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];         
        out.f = [out.f;sum(abs(Aub))];
        if (rel_chg_inn < tol) || numel(out.rel_chg_inn)>opts.iter, break; end    
    end
    
    % update multipliers
    deltap = delta;
    delta = delta - beta*(Aub-W);
    out.rdelta = [out.rdelta;norm(delta-deltap)/norm(deltap)];
    if out.rdelta(end) < tol, break; end

end
out.total_iter = numel(out.rel_chg_inn);
U = reshape(U,p,q,r);


function y = l1_l1_operator(A,D,Dt,u,mode,d1,mu)

% this function concatonates the two terms in the objective function
switch mode
    case 1
         y = mu*A(u,1);
         y = [y;col(D(u))];
    case 2
         y = mu*A(u(1:d1),2);
         y = y + Dt(u(d1+1:end));
end


