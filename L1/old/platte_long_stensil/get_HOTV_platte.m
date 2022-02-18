function [U,mu,beta,muf,betaf,muDbeta,...
    delta,gL,ind,out] = get_HOTV(p,q,r,Atb,AAtb,scl,opts,k,b,wrap_shrink)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences


% initialize U, beta, mu
muf = opts.mu;       % final mu
betaf = opts.beta;     % final beta
beta = opts.beta;
mu = opts.mu;
muDbeta = mu/beta;





% initialize D^T sigma + A^T delta
gL = zeros(p*q*r,1);



% declare out variables
out.rel_chg_inn = [];  out.rel_chg_out = [];
out.f = [];        % record values of augmented Lagrangian fnc
out.cnt = [];      % record # of back tracking
out.lam1 = []; out.lam2 = []; out.lam3 = []; out.lam4 = []; out.lam5 = [];
out.tau = []; out.alpha = []; out.C = []; out.mu = [];


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
if max([mm,nn,rr]) == 1
    % single gradient step least squares solution (according to my calculation)
    const = sum(b.*conj(AAtb))/max(sum(AAtb.*conj(AAtb)),eps);
    U = reshape(const*Atb,p,q,r);
%     switch opts.init
%         case 0, U = zeros(p,q,r);
%         case 1, U = reshape(Atb,p,q,r);
%     end
else
    if mm ~= p || nn ~= q || rr ~= r
        fprintf('Input initial guess has incompatible size! Switch to the default initial guess. \n');
        U = reshape(Atb,p,q,r);
    else
        U = opts.init*scl;
    end
end

% global D Dt
% [D,Dt] = get_D_Dt(k,p,q,r,opts,Atb);


% Check that Dt is the true adjoint of D


% initialize multiplers
                     
delta = zeros(length(b),1);


if ~wrap_shrink
    ind = get_ind(k,p,q,r);
else
    ind=[];
end