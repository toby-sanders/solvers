function [U, out] = HOTV3D(hhat,bhat,opts)

% written by Toby Sanders @Lickenbrock tech
% Last update: 12/2019

% HOTV L1 regularization and multiscale variants
% optimized using the ADMM
% this code has been unified to handle deconvolution and Fourier data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      deconvolution case 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs: 
%   hhat - 3D image stack of Fourier transforms of PSFs
%   bhat - 3D image stack of Fourier transforms of blurry image data
%   opts - options


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

% Outputs:
%   U: reconstructed signal
%   out: output numerics

[p,q,nF] = size(bhat);
if size(hhat,1)~=p || size(hhat,2)~=q || size(hhat,3)~=nF
    error('PSF and image data not compatible sizes');
end

opts = check_HOTV_opts(opts);  % get and check opts

% mark important variables
tol = opts.tol; 
tol_inn = max(tol,1e-5);  % inner loop tolerance isn't as important
k = opts.order;
[D,Dt] = FD2D(k,p,q);

% initialize solution, splitting variables and multipliers
U = zeros(p,q,1); 
sigma = D(U);  W = sigma; Uc = W;
% delta = zeros(length(b),1);
if size(opts.init,1) == p, U = opts.init; end
gL = zeros(p,q); % initialize gradient on lagrange multiplier

V = my_Fourier_filters(k,opts.levels,p,q,1);



% scaling operators and parameters for deconvolution
hhat2 = sum(abs(hhat).^2,3);
out.PSFscaling = max(hhat2(:));
hhat = hhat/sqrt(out.PSFscaling);
hhat2 = hhat2/out.PSFscaling;
bhat = bhat/sqrt(out.PSFscaling);
Atb = sum(ifft2(bhat.*conj(hhat)),3);



[~,out.paramsScaling] = Scaleb(col(ifft2(bhat)));%  Scaleb(b);
mu = opts.mu*out.paramsScaling; 
beta = opts.beta*out.paramsScaling;
params.gA = 0;
params.gD = 0;

if round(k)~=k || opts.levels>1 || k==0, opts.wrap_shrink = true; end
if ~opts.wrap_shrink, ind = get_ind(k,p,q,1);
    ind = ind(ind<p*q*2);
else, ind=[]; end



% initialize output variables
out.rel_chg_inn = []; out.objf_val = [];
ii = 0;  % main loop
while numel(out.rel_chg_inn) < opts.iter
    ii = ii + 1; % count number of multiplier updates
    updateMode = 'GD';
    for jj = 1:opts.inner_iter

        % alternating minimization steps between U and W
        [U,params] = updateU_HOTV_deblur(hhat2,hhat,Atb,D,Dt,U,Uc,W,gL,V,mu,beta,opts.nonneg,params,updateMode);
        Uc = D(U);
        W = shrinkage(Uc,opts.L1type,beta,sigma,opts.wrap_shrink,ind);
        if strcmp(updateMode,'GD')
            updateMode = 'BB';
        end

        % check for convergence
        rel_chg_inn = norm(params.uup)/norm(U(:));
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        if rel_chg_inn<opts.tol && ii>5
            out.mu = mu;
            out.total_iter = numel(out.rel_chg_inn);
            out.relUW = myrel(W,Uc);
            return;
        end

    end
    sigma = sigma - beta*(Uc-W);
    gL = Dt(sigma);%  + A(delta,2); 
end
out.mu = mu;
out.total_iter = numel(out.rel_chg_inn);
out.relUW = myrel(W,Uc);


function W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind)
% update step on W using shrinkage formulas
if strcmp(L1type,'anisotropic')
    W = Uc - sigma/beta;
    W = max(abs(W) - 1/beta, 0).*sign(W);    
elseif strcmp(L1type,'isotropic')
    W = Uc - sigma/beta;
    Ucbar_norm = sqrt(W(:,1).*conj(W(:,1)) + ...
        W(:,2).*conj(W(:,2)) );
    Ucbar_norm = max(Ucbar_norm - 1/beta, 0)./(Ucbar_norm+eps);
    W(:,1) = W(:,1).*Ucbar_norm;
    W(:,2) = W(:,2).*Ucbar_norm;
else
    error('Somethings gone wrong.  L1type is either isotropic or anisotropic');
end
% reset edge values if not using periodic regularization
if ~wrap_shrink, W(ind)=Uc(ind); end

