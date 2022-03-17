function [U,params] = updateU_HOTV(Shhat2,hhat,Atb,D,Dt,U,Uc,W,gL,V,mu,beta,nonneg,params,mode)

% single step updates on U for HOTV L1 ADMM optimization
% this is really just a quadratic minimization step



[p,q] = size(U);
Up = U; % save previous U for convergence checking and BB steps

switch mode
    case {'BB','GD'}
        if isfield(params,'gA')
            gAp = params.gA; 
            gDp = params.gD;
        end
        % compute gradient
        params.gD = Dt(Uc - W); % gD = D'*(Du-w)
        params.gA = ifft2(Shhat2.*fft2(U)) - Atb;% gA = A'(Au-b)
        g = beta*params.gD + mu*params.gA - gL; % complete gradient
        
        % determine step length
        if strcmp(mode,'BB')
            % BB-like step length
            dgA = params.gA - gAp;   
            dgD = params.gD - gDp;                    
            ss = params.uup'*params.uup;                      
            sy = params.uup'*(beta*dgD(:) + mu*dgA(:));       
            tau = abs(ss/max(sy,eps));   
        else
            % optimal step length at the 1st iteration
            gc = D(g);       
            denom1 = sum(col(gc.*conj(gc)));
            Ag = ifft2(fft2(g).*hhat);% A(g,1);
            denom2 = sum(col(Ag.*conj(Ag)));
            tau = (g(:)'*g(:))/(beta*denom1 + mu*denom2);
        end
        U = U - tau*g; % gradient descent
    case 'deconv'
        % updates for deconvolution
        % with large mu, this update will fail with nonneg constraint!
        alpha = .5;
        z = fft2(mu*Atb+reshape(Dt(beta*W)+gL,p,q,1));
        U = ifft2(z./(mu*Shhat2 + beta*V));
        U = alpha*U + (1-alpha)*Up;
end

% projected gradient method for inequality constraints
if nonneg
    U = max(real(U),0);
end
params.uup = U(:)-Up(:);
