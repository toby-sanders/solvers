function [U,params] = updateU_HOTV(A,D,Dt,U,Uc,W,gL,mu,beta,params,opts)

% written by Toby Sanders @Lickenbrock tech
% Last update: 12/2019

% single step updates on U for HOTV L1 ADMM optimization
% this is really just a quadratic minimization step
% The L1 norm is only involved in the shrinkage formula update on W
% For the general case, a gradient descent is used
% For deconvolution and Fourier data the exact minimizer is evaluated

[p,q,r] = size(U);
Up = U; % save previous U for convergence checking and BB steps
switch params.mode % handling various update types
    case {'BB','GD'} % gradient decent update on U
        if isfield(params,'gA')
            gAp = params.gA; 
            gDp = params.gD;
        end
        % compute gradient
        params.gD = Dt(Uc - W); % gD = D'*(Du-w)
        params.Au = A(U(:),1);
        params.gA = A(params.Au,2) - params.Atb;% gA = A'(Au-b)
        g = beta*params.gD + mu*params.gA - gL;
        
        % determine step length
        if strcmp(params.mode,'BB')
            % BB-like step length
            dgA = params.gA - gAp;   
            dgD = params.gD - gDp;                    
            ss = params.uup'*params.uup;                      
            sy = params.uup'*(beta*dgD + mu*dgA);       
            tau = abs(ss/max(sy,eps));   
        else
            % optimal step length at the 1st iteration
            gc = D(reshape(g,p,q,r));       
            dDd = sum(col(gc.*conj(gc)));
            Ag = A(g,1);
            tau = abs((g'*g)/(beta*dDd + mu*(Ag')*Ag));
            params.mode = 'BB'; % do BB step length for next update
        end
        U = U - tau*reshape(g,p,q,r); % gradient descent
    case 'Fourier' % updates for Fourier data
        bb = fftn(mu*params.ibstr+reshape(Dt(beta*W)+gL,p,q,r))/sqrt(p*q*r);
        params.Au = bb./(mu*params.VS + beta*params.V);
        U = ifftn(params.Au)*sqrt(p*q*r);
        params.Au = params.Au(A);
    case 'deconv' % updates for deconvolution 
        bb = fftn(mu*params.Atb+reshape(Dt(beta*W)+gL,p,q,r));
        U = ifftn(bb./(mu*A + beta*params.V));
end
% projected gradient method for inequality constraints
if opts.nonneg, U = max(real(U),0);
elseif opts.isreal, U = real(U); end
if opts.max_c, U = min(U,opts.max_v); end
params.uup = U(:)-Up(:);
