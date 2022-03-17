function [U, out] = HOTV2D_MFD(hhat,bhat,opts)

% written by Toby Sanders @magnetic insight

% multiframe deconvolution with L1 regularization
% optimized using the ADMM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      multiframe deconvolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inputs: 
%   hhat - 3D image stack of Fourier transforms of PSFs
%   bhat - 3D image stack of Fourier transforms of blurry image data
%   opts - options

% Outputs:
%   U: deconvolved image
%   out: output numerics

[p,q,nF] = size(bhat);
if size(hhat,1)~=p || size(hhat,2)~=q || size(hhat,3)~=nF
    error('PSF and image data not compatible sizes');
end
opts = check_HOTV_opts(opts);  % get and check opts

% mark important variables
tol = opts.tol; 
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

% scale the parameters to account for data scaling and L1 vs. L2 norms
[~,out.paramsScaling] = Scaleb(col(ifft2(bhat)));
mu = opts.mu*out.paramsScaling; 
beta = opts.beta*out.paramsScaling;
params.gA = 0;
params.gD = 0;
out.rel_chg_inn = []; out.objf_val = []; % output variables

if round(k)~=k || opts.levels>1 || k==0, opts.wrap_shrink = true; end
if ~opts.wrap_shrink, ind = get_ind(k,p,q,1);
    ind = ind(ind<p*q*2);
else, ind=[]; end

% set up the update function so only pass in parameters that change
LocalUpdateU = @(U,Uc,W,gL,mu,beta,params,updateMode)updateU_MFD(...
    hhat2,hhat,Atb,D,Dt,U,Uc,W,gL,V,mu,beta,opts.nonneg,params,updateMode);

ii = 0;  % main loop
while numel(out.rel_chg_inn) < opts.iter
    ii = ii + 1; % count number of multiplier updates
    updateMode = 'GD';
    for jj = 1:opts.inner_iter
        % alternating minimization steps between U and W
        [U,params] = LocalUpdateU(U,Uc,W,gL,mu,beta,params,updateMode);
        Uc = D(U);
        W = shrinkage(Uc,opts.L1type,beta,sigma,opts.wrap_shrink,ind);
        if strcmp(updateMode,'GD')
            updateMode = 'BB';
        end

        % check for convergence
        rel_chg_inn = norm(params.uup)/norm(U(:));
        out.rel_chg_inn = [out.rel_chg_inn;rel_chg_inn];
        if rel_chg_inn<opts.tol && ii>5 
            out.mu = mu; % if converged, output vars and return
            out.total_iter = numel(out.rel_chg_inn);
            out.relUW = myrel(W,Uc);
            return;
        end
    end
    % update Lagrange multiplier and its gradient
    sigma = sigma - beta*(Uc-W);
    gL = Dt(sigma);
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

