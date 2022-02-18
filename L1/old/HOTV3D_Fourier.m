function [U, out] = HOTV3D_Fourier(S,b,n,opts)


% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% Last update: 05/17/2018
%
% Fast, signal, image, and video Fourier reconstruction with HOTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Problem Description       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [U, out] = HOTV3D_denoise(b,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||PFu - b||_2^2 + ||D^k u||_1 }
%
% where D^k is kth order finite difference, F is DFT.
% P is a row selector matrix that selects the known coefficients in b
% Multiscale finite differences D^k can also be used.
% To see how to modify these settings read the file "check_HOTV_opts.m"
%
% The problem is modified using variable splitting and this algorithm 
% works with the following augmented Lagrangian function: 
%
%      min_{u,w} {mu/2 ||PFu - b||_2^2 + beta/2 ||D^k u - w ||_2^2 
%               + ||w||_1 - (sigma , D^k u - w) }
%
% sigma is a Lagrange multiplier
% Algorithm uses alternating direction minimization over u and w.
%
% Inputs: 
%   S: indices of input Fourier coefficients
%   b: Fourier coefficients
%   n: signal dimension(s)
%   opts: structure containing input parameters, 
%       see function check_HOTV_opts.m for these
%
% Outputs:
%   U: reconstructed signal/image
%   out: output numerics
warning('this code does not appear correct for higher orders');
tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
opts = check_HOTV_opts(opts);  % get and check opts
tol = opts.tol; 
tol_inn = max(tol,1e-4);  % inner loop tolerance isn't as important
k = opts.order; wrap_shrink = opts.wrap_shrink;
L1type = opts.L1type; nlev = opts.levels;
[~,scl] = Scaleb(b);
beta = opts.beta*scl; mu = opts.mu;
if round(k)~=k || opts.levels>1, wrap_shrink = true; end
if ~wrap_shrink, ind = get_ind(k,p,q,r);
else, ind=[]; end
[D,Dt] = get_D_Dt(k,p,q,r,opts);
flg = check_D_Dt(D,Dt,[p,q,r]);
if ~flg, error('Sparse domain transforms do not appear consistent'); end
U = zeros(p,q,r); 
sigma = D(U);  % initialize multipler
W = sigma;
nrmb = norm(b);
V = get_myFilters(k,nlev,p,q,r);
VS = zeros(p,q,r);VS(S) = 1;

out.rel_chg = [];
ii = 0; iter = 0;
bstr = zeros(p,q,r);
bstr(S) = b;
ibstr = ifftn(bstr)*sqrt(prod(n));
while iter <= opts.iter
    ii = ii + 1;    
    for jj = 1:opts.inner_iter
        iter = iter + 1;
        Up = U;
        bb = fftn(mu*ibstr+reshape(Dt(beta*W + sigma),p,q,r))/sqrt(prod(n));
        U = ifftn(bb./(mu*VS + beta*V))*sqrt(prod(n));
        % figure(96);plot(abs(U));title([num2str(ii),' ',num2str(iter)]);pause;

        % projection method and shrinkage step
        if opts.nonneg, U = max(real(U),0);
        elseif opts.isreal, U = real(U); end
        if opts.max_c, U = min(U,opts.max_v); end        
        U = reshape(U,p,q,r);  Uc = D(U);
        W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind);
        
        % store rel change, check inner loop convergence
        uup = U - Up; uup = uup(:);
        rel_chg = norm(uup)/norm(Up(:));
        out.rel_chg = [out.rel_chg;rel_chg];
        if opts.store_soln, out.Uall(:,:,jj+(ii-1)*opts.inner_iter) = U; end      
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            fprintf('iter=%i, rel_chg_sol=%10.4e, ||Du-w||^2=%10.4e\n',...
                iter, rel_chg, norm(col(Uc-W))^2);
        end       
        if (rel_chg < tol_inn) || iter>opts.iter, break; end 
    end
    % my convergence criteria
    if jj<3 && rel_chg < tol && ii>4, break;end
    
    % end of inner loop, update Lagrange multiplier
    sigma = update_mlp(beta,W,Uc,sigma);
    if opts.disp, fprintf(' Lagrange mlps update number %i\n',ii);end
end

out.final_error = norm(fftn(U)-bstr)/nrmb;
% output these so one may check optimality conditions
out.optimallity = sigma + beta*(W-Uc); out.W = W;
tmp = fftn(U)/sqrt(prod(n));tmp2 = zeros(p,q,r);tmp2(S) = tmp(S);
out.optimallity2 = col(mu*sqrt(prod(n))*ifftn(tmp2-bstr))...
    + Dt(beta*(D(U)-W) -sigma);
out.elapsed_time = toc;

function sigma = update_mlp(beta,W,Uc,sigma)
V = Uc - W;
sigma = sigma - beta*V;

function W = shrinkage(Uc,L1type,beta,sigma,wrap_shrink,ind)

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
    error('Somethings wrong.  L1type is either isotropic or anisotropic');
end

% reset edge values if not using periodic regularization
if ~wrap_shrink, W(ind)=Uc(ind); end


function V = get_myFilters(k,nlev,p,q,r)

% eigenvalues of regularization operator which act as filters in Fourier
if k~=0
vx = zeros(nlev,q); vy = zeros(nlev,p); vz = zeros(nlev,r);
for i = 1:nlev
    vx(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,q-1,q)/q).^(2*k+2))./...
            (sin(pi*linspace(0,q-1,q)/q).^2)*(2^(2-k-i)/nlev)^2;
    vy(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,p-1,p)/p).^(2*k+2))./...
        (sin(pi*linspace(0,p-1,p)/p).^2)*(2^(2-k-i)/nlev)^2;
    vz(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,r-1,r)/r).^(2*k+2))./...
        (sin(pi*linspace(0,r-1,r)/r).^2)*(2^(2-k-i)/nlev)^2;
end
vx(:,1) = 0; vy(:,1) = 0; vz(:,1) = 0;
vx = sum(vx,1); vy = sum(vy,1); vz = sum(vz,1); 
V = vy' + vx + reshape(vz,1,1,r);
else
V = 4*ones(p,q,r);
end