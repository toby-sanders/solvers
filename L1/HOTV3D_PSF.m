function [U, out] = HOTV3D(A,b,n,opts)

% written by Toby Sanders @Lickenbrock tech
% Last update: 12/2019

% HOTV L1 regularization and multiscale variants
% optimized using the ADMM
% this code has been unified to handle deconvolution and Fourier data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      deconvolution case 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs: 
%   A - blurring kernel, same size as b
%   b - blurry image data
%   opts.mode = 'deconv'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Fourier case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs:
%   A - indices of acquired Fourier coefficients
%   b - values of acquired Fourier coefficients
%   opts.mode - 'Fourier'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       general case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs:
%   A - forward and adjoint operator
%   b - data of form b = A*u + epsilon
%   opts.mode - empty or GD or BB


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   General Problem Description     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [U, out] = HOTV3D(A,b,n,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||Au - b||_2^2 + ||D^k u||_1 }
%
% where D^k is kth order finite difference.
% Multiscale finite differences D^k can also be used.
% To see how to modify these settings read the file "check_HOTV_opts.m"
%
% The problem is modified using variable splitting and this algorithm 
% works with the following augmented Lagrangian function: 
%
%      min_{u,w} {mu/2 ||Au - b||_2^2 + beta/2 ||D^k u - w ||_2^2 
%               + ||w||_1 - (delta , Au - b ) - (sigma , D^k u - w) }
%
% delta and sigma are Lagrange multipliers
% Algorithm uses alternating direction minimization over u and w.
%
% delta has actually been removed.  If one wants to solve the constrained
% minimization problem, Au = b, then it will need to be modified to do so.
%
% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   b: data values in vector form
%   n: image/ signal dimensions in vector format
%   opts: structure containing input parameters, 
%       see function check_HOTV_opts.m for these
%
% Outputs:
%   U: reconstructed signal
%   out: output numerics
opts.mode = 'deconv';
if ~isfield(opts,'supportWidth'), sWx = 10; sWy = 10;
else
    if numel(opts.supportWidth)==1
        opts.supportWidth = opts.supportWidth*ones(2,1);
    end
    sWx = opts.supportWidth(2);
    sWy = opts.supportWidth(1);
end


tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
opts = check_HOTV_opts(opts);  % get and check opts

% mark important variables
tol = opts.tol; 
tol_inn = max(tol,1e-5);  % inner loop tolerance isn't as important
k = opts.order; n = p*q*r;
[D,Dt] = get_D_Dt(k,p,q,r,opts);

% initialize solution, splitting variables and multipliers
U = zeros(p,q,r); 
sigma = D(U);  W = sigma; Uc = W;
% delta = zeros(length(b),1);
if size(opts.init,1) == p, U = opts.init; end
gL = zeros(p*q*r,1); % initialize gradient on lagrange multiplier

% For each case, initialize variables used in code that are not updated
if ~sum(strcmp(opts.mode,{'Fourier','deconv'}))
    % check that A* is true adjoint of A
    % check scaling of parameters, etc.
    if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
    [flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
    if ~flg, error('A and A* do not appear consistent'); end; clear flg;
    if opts.scale_A, [A,b] = ScaleA(n,A,b); end
    params.Atb = A(b,2); % A'*b
    [~,scl] = Scaleb(b);
    if opts.scale_mu, opts.mu = opts.mu*scl; end
    opts.beta = opts.beta*scl;
elseif strcmp(opts.mode,'deconv')
    params.V = my_Fourier_filters(k,opts.levels,p,q,r);
    A = fftn(A); % overwrite convolution kernel with its FT
    params.Atb = ifftn(fftn(b).*conj(A));
    A = A.*conj(A); 
    
else % Fourier case
    warning('this Fourier version of the code does not work for higher orders');
    params.ibstr = zeros(p,q,r);
    params.ibstr(A) = b;
    params.ibstr = ifftn(params.ibstr)*sqrt(prod(n));
    params.V = my_Fourier_filters(k,opts.levels,p,q,r);
    params.VS = zeros(p,q,r);
    params.VS(A) = 1;
end

if round(k)~=k || opts.levels>1, opts.wrap_shrink = true; end
if ~opts.wrap_shrink, ind = get_ind(k,p,q,r);
else, ind=[]; end

% initialize everything else
out.rel_chg_inn = []; out.objf_val = [];
beta = opts.beta;
mu = opts.mu;

objf_best = 1e20;
ii = 0;  % main loop








while numel(out.rel_chg_inn) < opts.iter
    ii = ii + 1; % count number of multiplier updates
    % set the update mode for U
    params.mode = 'GD'; % first step after each update is optimal descent
    if strcmp(opts.mode,'Fourier'), params.mode = 'Fourier';
    elseif strcmp(opts.mode,'deconv'), params.mode = 'deconv'; 
    end
    for jj = 1:opts.inner_iter
        % alternating minimization steps between U and W
        [U,params] = updateU_HOTV(A,D,Dt,U,Uc,W,gL,mu,beta,params,opts);
        
        % implement sum condition, image/PSF should sum to 1
        [S1,S2] = sort(U(:));
        ss = 0;
        ssCnt = numel(U);
        while ss<1
            ss = ss + S1(ssCnt);
            ssCnt = ssCnt - 1;
        end
        U(S2(1:ssCnt)) = 0;
        
        % shrinkage!!
        Uc = D(U);
        W = shrinkage(Uc,opts.L1type,beta,sigma,opts.wrap_shrink,ind);
        
        % check for convergence
        rel_chg_inn = norm(params.uup)/norm(U(:));
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        
        %  compute objective function values for general case
        if isfield(params,'Au')
            Aub = params.Au - b;
            out.objf_val = [out.objf_val; mu/2*(Aub)'*Aub+sum(abs(Uc(:)))]; 
            if out.objf_val(end) < objf_best
                objf_best = out.objf_val(end);out.Ubest = U;
                out.Ubest_iter = numel(out.rel_chg_inn)-1;
            end
        end      
        % check inner loop convergence
        if (rel_chg_inn < tol_inn) || numel(out.rel_chg_inn)>=opts.iter, break; end 

    end
    % convergence criteria
    if jj<opts.inner_iter && rel_chg_inn<tol && ii>4, break;end
    
    % update multipliers and gradient
    % if opts.data_mlp, delta = delta - mu*Aub; end
    sigma = sigma - beta*(Uc-W);
    gL = Dt(sigma);%  + A(delta,2);    
end

out.total_iter = numel(out.rel_chg_inn);
if isfield(params,'Au')
    out.final_error = norm(params.Au-b)/norm(b(:));
    fprintf('||Au-b||/||b||: %5.3f\n',out.final_error);
    fprintf('mu/2*||Au-b||^2 + ||Du||_1: %g\n',out.objf_val(end));
end
% output these so one may check optimality conditions
out.optimallity = sigma + beta*(W-Uc);
if ~sum(strcmp(opts.mode,{'Fourier','deconv'}))
out.optimallity2 = mu*params.gA + beta*params.gD - gL;
end
out.mu = mu;
out.elapsed_time = toc;
fprintf('total iteration count: %i\n',out.total_iter);

function W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind)
% update step on W using shrinkage formulas
if strcmp(L1type,'anisotropic')
    W = Uc - sigma/beta;
    W = max(abs(W) - 1/beta, 0).*sign(W);    
elseif strcmp(L1type,'isotropic')
    W = Uc - sigma/beta;
    Ucbar_norm = sqrt(W(:,1).*conj(W(:,1)) + ...
        W(:,2).*conj(W(:,2)) + W(:,3).*conj(W(:,3)));
    Ucbar_norm = max(Ucbar_norm - 1/beta, 0)./(Ucbar_norm+eps);
    W(:,1) = W(:,1).*Ucbar_norm;
    W(:,2) = W(:,2).*Ucbar_norm;
    W(:,3) = W(:,3).*Ucbar_norm;
else
    error('Somethings gone wrong.  L1type is either isotropic or anisotropic');
end
% reset edge values if not using periodic regularization
if ~wrap_shrink, W(ind)=Uc(ind); end

