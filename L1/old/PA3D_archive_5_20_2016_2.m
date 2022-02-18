function [U, out] = PA3D(A,b,p,q,r,opts)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code has been modified to solve l1 penalty problems with the
% polynomial annihilation transform
% Edited by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016

% original code can be found here: 
% http://www.caam.rice.edu/~optimization/L1/TVAL3/



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original algorithm description begins here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Goal: solve   min sum ||D_i u||_1
%                  s.t. Au = b
%       to recover video u from encoded b,
%       which is equivalent to solve       min sum ||w_i||
%                                          s.t. D_i u = w_i
%                                               Au = b
% Here we use 3D anisotropic finite difference operation as objective function.
%
% poly_ann_3D solves the corresponding augmented Lagrangian function:
%
% min_{u,w} sum ( ||w_i||_1 - sigma'(D_i u - w_i) + beta/2||D_i u - w_i||_2^2 )
%                   - delta'(Au-b) + mu/2||Au-b||_2^2 ,
%
% by an alternating algorithm:
% i)  while not converge
%     1) Fix w^k, do Gradient Descent to
%            - sigma'(Du-w^k) - delta'(Au-b) + beta/2||Du-w^k||^2 + mu/2||Au-f||^2;
%            u^k+1 is determined in the following way:
%         a) compute step length tau > 0 by BB formula
%         b) determine u^k+1 by
%                  u^k+1 = u^k - alpha*g^k,
%            where g^k = -D'sigma - A'delta + beta D'(Du^k - w^k) + mu A'(Au^k-f),
%            and alpha is determined by Amijo-like nonmonotone line search;
%     2) Given u^k+1, compute w^k+1 by shrinkage
%                 w^k+1 = shrink(Du^k+1-sigma/beta, 1/beta);
%     end
% ii) update Lagrangian multipliers by
%             sigma^k+1 = sigma^k - beta(Du^k+1 - w^k+1)
%             delta^k+1 = delta^k - mu(Au^k+1 - b).
% iii)accept current u as the initial guess to run the loop again
%
% Inputs:
%       A        : either an matrix representing the measurement or a struct
%                  with 2 function handles:
%                           A(x,1) defines @(x) A*x;
%                           A(x,2) defines @(x) A'*x;
%       b        :  input vector representing the compressed
%                   observation of a grayscale video
%       p, q     :  resolution
%       r        :  # of frames
%       opts     :  structure to restore parameters
%
%
% variables in this code:
%
% lam1 = sum ||wi||_1
% lam2 = ||Du-w||^2 (at current w).
% lam3 = ||Au-f||^2
% lam4 = sigma'(Du-w)
% lam5 = delta'(Au-b)
%
%   f  = lam1 + beta/2 lam2 + mu/2 lam3 - lam4 - lam5
%
%   g  = A'(Au-f)
%   g2 = D'(Du-w) (coefficients beta and mu are not included)
%
%
%
%
% Written by: Chengbo Li @ Bell Laboratories, Alcatel-Lucent
% Computational and Applied Mathematics department, Rice University
% 06/18/2010

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original algorithm description ends here   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end

% get or check opts
opts = check_PA_opts(opts);
ko = opts.order;
% cannot reset edge value without integer FD scheme
if round(ko)==ko
    wrap_shrink = opts.wrap_shrink;
else
    wrap_shrink = true;
end



% problem dimension
n = p*q*r;




% mark important constants
tol_inn = opts.tol_inn;
tol_out = opts.tol_out;




% check if A*A'=I
tmp = rand(length(b),1);
if norm(A(A(tmp,2),1)-tmp,1)/norm(tmp,1) < 1e-3
    opts.scale_A = false;
end
clear tmp;


% check scaling A
if opts.scale_A
    [A,b] = ScaleA(n,A,b);
end

% check scaling b
scl = 1;
if opts.scale_b
    [b,scl] = Scaleb(b);
end

% check for maximum constraint value
if opts.max_c
    max_v = opts.max_v*scl;
end


% calculate A'*b
Atb = A(b,2);


global D

[U,mu,beta,muf,betaf,muDbeta,sigmax,sigmay,sigmaz,delta,DtsAtd,out] ...
    = init_U(p,q,r,Atb,scl,opts,ko,b);    % U: p*q

nrmb = norm(b);
Upout = U;
[Ux,Uy,Uz] = D(U);                   % Ux, Uy, Uz: p*q*z

% first shrinkage step
Wx = max(abs(Ux) - 1/beta, 0).*sign(Ux);
Wy = max(abs(Uy) - 1/beta, 0).*sign(Uy);
Wz = max(abs(Uz) - 1/beta, 0).*sign(Uz);

% reset edge values if not using periodic regularization
if ~wrap_shrink
    if ko<=q, Wx(:,end-ko+1:end,:) = Ux(:,end-ko+1:end,:); end;
    if ko<=p, Wy(end-ko+1:end,:,:) = Uy(end-ko+1:end,:,:); end;
    if ko<=r, Wz(:,:,end-ko+1:end) = Uz(:,:,end-ko+1:end); end;
end

lam1 = sum(sum(sum(abs(Wx) + abs(Wy) + abs(Wz))));

% g and gD are the gradients of ||Au-b||^2 and ||Du-w||^2, respectively
[lam2,lam3,lam4,lam5,f,gD,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,...
    lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz,delta);


% compute gradient
d = gD + muDbeta*g - DtsAtd;


out.f = [out.f; f]; 
out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; out.lam3 = [out.lam3; lam3];
out.lam4 = [out.lam4; lam4]; out.lam5 = [out.lam5; lam5];out.mu = [out.mu; mu];

        
for ii = 1:opts.outer_iter
    if opts.disp
            fprintf('    Beginning outer iteration #%d\n',ii);
            fprintf('    mu = %d , beta = %d , order = %g\n',mu,beta,ko);
            fprintf('iter    ||w||_1    ||Du - w||^2  ||Au - b||^2\n');
    end
        
    %initialize the constants
    gam = opts.gam; Q = 1; fp = f;
    
    
    for jj = 1:opts.inner_iter
        % compute step length, tau
        if jj~=1
            dg = g - gp;                        % dg: pq
            dgD = gD - gDp;                     % dgD: pq
            ss = uup'*uup;                      % ss: constant
            sy = uup'*(dgD + muDbeta*dg);       % sy: constant
            % compute BB step length
            tau = abs(ss/max(sy,eps));          % tau: constant
        else
            % do Steepest Descent at the 1st ieration
            %d = gD + muDbeta*g - DtsAtd;         % d: pq
            [dx,dy,dz] = D(reshape(d,p,q,r));       
            % dDd: cosntant
            dDd = sum(sum(sum(dx.*conj(dx)+dy.*conj(dy) + dz.*conj(dz))));
            Ad = A(d,1);                        %Ad: m
            % compute Steepest Descent step length
            tau = abs((d'*d)/(dDd + muDbeta*Ad'*Ad));
        end

        % keep the previous values for backtracking
        Up = U; gp = g; gDp = gD; Aup = Au; 
        Uxp = Ux; Uyp = Uy; Uzp = Uz; DtsAtdp =  DtsAtd;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ONE-STEP GRADIENT DESCENT %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = U(:) - tau*d;
        % projected gradient method for nonnegtivity
        if opts.nonneg
            U = max(real(U),0);
        elseif opts.isreal
            U = real(U);
        end
        if opts.max_c
            U = min(U,max_v);
        end
        U = reshape(U,p,q,r);
        [Ux,Uy,Uz] = D(U);

        [lam2,lam3,lam4,lam5,f,gD,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,...
            lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz,delta);

        % Nonmonotone Line Search
        

        % Unew = Up + alpha*(U - Up)
        % f should be decreasing, if not, then the algorithm steps moves U
        % back in the direction of the previous solution
        alpha = 1;
        du = U - Up;
        const = 1e-5*beta*(d'*d*tau);
        cnt = 0; flg = true;
        while f > fp - alpha*const
            if cnt <5
                if flg
                    dg = g - gp;
                    dgD = gD - gDp;
                    dAu = Au - Aup;
                    dUx = Ux - Uxp;
                    dUy = Uy - Uyp;
                    dUz = Uz - Uzp;
                    flg = false;
                end
                % shrink alpha
                alpha = alpha*opts.gamma;
                % U is moved back toward Up, in particular: 
                % U = alpha*U +(1-alpha)Up;
                [U,lam2,lam3,lam4,lam5,f,Ux,Uy,Uz,Au,g,gD] = update_g(p,q,r,...
                    lam1,alpha,beta,mu,Up,du,gp,dg,gDp,dgD,Aup,dAu,Wx,Wy,Wz,...
                    Uxp,dUx,Uyp,dUy,Uzp,dUz,b,sigmax,sigmay,sigmaz,delta);
                cnt = cnt + 1;
            else
                
                % shrink gam
                gam = opts.rate_gam*gam;

                % give up and take Steepest Descent step
                if (opts.disp > 0) && (mod(jj,opts.disp) == 0)
                    disp('    count of back tracking attains 5 ');
                end

                %d = gDp + muDbeta*gp - DtsAtd;
                [dx,dy,dz] = D(reshape(d,p,q,r));
                dDd = sum(sum(sum(...
                    dx.*conj(dx)+dy.*conj(dy) + dz.*conj(dz))));
                Ad = A(d,1);
                tau = abs((d'*d)/(dDd + muDbeta*Ad'*Ad));
                U = Up(:) - tau*d;
                % projected gradient method for nonnegtivity
                if opts.nonneg
                    U = max(real(U),0);
                elseif opts.isreal
                    U = real(U);
                end
                
                U = reshape(U,p,q,r);
                [Ux,Uy,Uz] = D(U);
                Uxbar = Ux - sigmax/beta;
                Uybar = Uy - sigmay/beta;
                Uzbar = Uz - sigmaz/beta;
                Wx = max(abs(Uxbar) - 1/beta, 0).*sign(Uxbar);
                Wy = max(abs(Uybar) - 1/beta, 0).*sign(Uybar);
                Wz = max(abs(Uzbar) - 1/beta, 0).*sign(Uzbar);
                % reset edge values if not using periodic regularization
                if ~wrap_shrink
                    if ko<=q, Wx(:,end-ko+1:end,:) = Ux(:,end-ko+1:end,:); end;
                    if ko<=p, Wy(end-ko+1:end,:,:) = Uy(end-ko+1:end,:,:); end;
                    if ko<=r, Wz(:,:,end-ko+1:end) = Uz(:,:,end-ko+1:end); end;
                end

                lam1 = sum(sum(sum(abs(Wx) + abs(Wy) + abs(Wz))));
                [lam2,lam3,lam4,lam5,f,gD,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,...
                    lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz,delta);
                alpha = 0; % remark the failure of back tracking
                break;
            end
            
        end
        


        % if back tracking is successful, then recompute
        if alpha ~= 0
            Uxbar = Ux - sigmax/beta;
            Uybar = Uy - sigmay/beta;
            Uzbar = Uz - sigmaz/beta;
            Wx = max(abs(Uxbar) - 1/beta, 0).*sign(Uxbar);
            Wy = max(abs(Uybar) - 1/beta, 0).*sign(Uybar);
            Wz = max(abs(Uzbar) - 1/beta, 0).*sign(Uzbar);

            % reset edge values if not using periodic regularization
            if ~wrap_shrink
                if ko<=q, Wx(:,end-ko+1:end,:) = Ux(:,end-ko+1:end,:); end;
                if ko<=p, Wy(end-ko+1:end,:,:) = Uy(end-ko+1:end,:,:); end;
                if ko<=r, Wz(:,:,end-ko+1:end) = Uz(:,:,end-ko+1:end); end;
            end


            % update parameters related to Wx, Wy
            [lam1,lam2,lam4,f,gD] = update_W(beta,...
                Wx,Wy,Wz,Ux,Uy,Uz,sigmax,sigmay,sigmaz,lam1,lam2,lam4,f);
        end

        % update reference value
        Qp = Q; Q = gam*Qp + 1; fp = (gam*Qp*fp + f)/Q;
        uup = U - Up; uup = uup(:);           % uup: pq

        out.f = [out.f; f]; out.C = [out.C; fp]; out.cnt = [out.cnt;cnt];
        out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; out.lam3 = [out.lam3; lam3];
        out.lam4 = [out.lam4; lam4]; out.lam5 = [out.lam5; lam5];
        out.tau = [out.tau; tau]; out.alpha = [out.alpha; alpha];out.mu = [out.mu; mu];

        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            prnt_format = '%3.0f %10.5g %12.5g %13.5g\n';
            fprintf(prnt_format, jj,lam1,lam2,lam3);
        end


        % recompute gradient
        d = gD + muDbeta*g - DtsAtd;
        
        % move to next outer iteration and update multipliers if relative
        % change is less than tolerance
        if (norm(d) < tol_inn), break; end;
        
        
    end
    
    
    RelChgOut = norm(U(:)-Upout(:))/norm(Upout(:));
    out.reer = [out.reer; RelChgOut];
    Upout = U;

    % stop if already reached optimal solution
    if RelChgOut < tol_out || sqrt(lam3(end))/nrmb<opts.min_l2_error
        break;
    end

    % update multipliers
    [sigmax,sigmay,sigmaz,delta,lam4,lam5] = update_mlp(beta,mu, ...
        Wx,Wy,Wz,Ux,Uy,Uz,Au,b,sigmax,sigmay,sigmaz,delta);
    if ~opts.data_mlp
        delta(:) = 0;
        lam5 = 0;
    end


    % update penality parameters for continuation scheme
    beta0 = beta;
    beta = min(betaf, beta*opts.rate_ctn);
    mu = min(muf, mu*opts.rate_ctn);
    muDbeta = mu/beta;

    % update function value, gradient, and relavent constant
    f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;
    DtsAtd = -(beta0/beta)*d;     % DtsAtd should be divided by new beta instead of the old one for consistency!  
    d = gD + muDbeta*g - DtsAtd;

        
        
     
        

end

if sqrt(lam3(end))/nrmb<opts.min_l2_error
    fprintf('\nREACHED OPTIMAL L2 ERROR!!!\n\n');
end
out.total_iter = jj*ii;
out.final_error = norm(A(U(:),1)-b)/nrmb;
out.final_wl1 = lam1(end);
out.final_Du_w = lam2(end);
final_disp(out,opts);
            
% rescale U
U = U/scl;





function [lam2,lam3,lam4,lam5,f,gD,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,...
    lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz,delta)
global Dt

% A*u
Au = A(U(:),1);

% g 
g = A(Au,2) - Atb;



% lam2, ||Du - w||^2
Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
%lam2 = sum(sum(sum(Vx.*Vx + Vy.*Vy + Vz.*Vz)));
lam2 = sum(sum(sum(Vx.*conj(Vx) + Vy.*conj(Vy) + Vz.*conj(Vz))));

% gD = D'(Du-w)
gD = Dt(Vx,Vy,Vz);


% lam3
Aub = Au-b;
lam3 = norm(Aub)^2;

%lam4
lam4 = sum(sum(sum(sigmax.*Vx + sigmay.*Vy + sigmaz.*Vz)));

%lam5
lam5 = delta'*Aub;

% f
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;



function [U,lam2,lam3,lam4,lam5,f,Ux,Uy,Uz,Au,g,gD] = update_g(p,q,r,lam1,...
    alpha,beta,mu,Up,du,gp,dg,gDp,dgD,Aup,dAu,Wx,Wy,Wz,Uxp,dUx,Uyp,dUy,...
    Uzp,dUz,b,sigmax,sigmay,sigmaz,delta)

g = gp + alpha*dg;
gD = gDp + alpha*dgD;
U = Up + alpha*reshape(du,p,q,r);
Au = Aup + alpha*dAu;
Ux = Uxp + alpha*dUx;
Uy = Uyp + alpha*dUy;
Uz = Uzp + alpha*dUz;

Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;

lam2 = sum(sum(sum(Vx.*conj(Vx) + Vy.*conj(Vy) + Vz.*conj(Vz))));
Aub = Au-b;
lam3 = norm(Aub)^2;
lam4 = sum(sum(sum(sigmax.*Vx + sigmay.*Vy + sigmaz.*Vz)));
lam5 = delta'*Aub;
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;



function [lam1,lam2,lam4,f,gD] = update_W(beta,...
    Wx,Wy,Wz,Ux,Uy,Uz,sigmax,sigmay,sigmaz,lam1,lam2,lam4,f)
global Dt

% update parameters because Wx, Wy were updated
tmpf = f -lam1 - beta/2*lam2 + lam4;
lam1 = sum(sum(sum(abs(Wx) + abs(Wy) + abs(Wz))));
Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
gD = Dt(Vx,Vy,Vz);
lam2 = sum(sum(sum(Vx.*conj(Vx) + Vy.*conj(Vy) + Vz.*conj(Vz))));
lam4 = sum(sum(sum(sigmax.*Vx + sigmay.*Vy + sigmaz.*Vz)));
f = tmpf +lam1 + beta/2*lam2 - lam4;



function [sigmax,sigmay,sigmaz,delta,lam4,lam5] = update_mlp(beta,mu, ...
    Wx,Wy,Wz,Ux,Uy,Uz,Au,b,sigmax,sigmay,sigmaz,delta)


Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
sigmax = sigmax - beta*Vx;
sigmay = sigmay - beta*Vy;
sigmaz = sigmaz - beta*Vz;
Aub = Au-b;
delta = delta - mu*Aub;

%tmpf = f + lam4 + lam5;
lam4 = sum(sum(sum(sigmax.*Vx + sigmay.*Vy + sigmaz.*Vz)));
lam5 = delta'*Aub;
%f = tmpf - lam4 - lam5;







function [U,mu,beta,muf,betaf,muDbeta,sigmax,sigmay,sigmaz,...
    delta,DtsAtd,out] = init_U(p,q,r,Atb,scl,opts,ko,b)

% initialize U, beta, mu
muf = opts.mu;       % final mu
betaf = opts.beta;     % final beta
beta = opts.beta0;
mu = opts.mu0;
muDbeta = mu/beta;



% initialize multiplers
sigmax = zeros(p,q,r);                       
sigmay = zeros(p,q,r);
sigmaz = zeros(p,q,r);
delta = zeros(length(b),1);

% initialize D^T sigma + A^T delta
DtsAtd = zeros(p*q*r,1);



% prepare for iterations
out.reer = [];     % record relative errors of outer iterations
out.f = [];        % record values of augmented Lagrangian fnc
out.cnt = [];      % record # of back tracking
out.lam1 = []; out.lam2 = []; out.lam3 = []; out.lam4 = []; out.lam5 = [];
out.tau = []; out.alpha = []; out.C = []; out.mu = [];


% initialize U
[mm,nn,rr] = size(opts.init);
if max([mm,nn,rr]) == 1
    switch opts.init
        case 0, U = zeros(p,q,r);
        case 1, U = reshape(Atb,p,q,r);
    end
else
    if mm ~= p || nn ~= q || rr ~= r
        fprintf('Input initial guess has incompatible size! Switch to the default initial guess. \n');
        U = reshape(Atb,p,q,r);
    else
        U = opts.init*scl;
    end
end

global D Dt
if opts.smooth_phase
    if round(ko) == ko
        [D,Dt] = FD3D(ko);
    else
        [D,Dt] = FFD3D(ko,p,q,r); 
    end
else
    if sum(sum(sum(abs(opts.phase_angles))))==0
        opts.phase_angles = exp(-1i*reshape(Atb,p,q,r));
    else
        opts.phase_angles = exp(-1i*opts.phase_angles);
    end
    if round(ko) == ko
        [D,Dt] = FD3D_complex(ko,opts.phase_angles);
    else
        [D,Dt] = FFD3D_complex(ko,p,q,r,opts.phase_angles);
    end
end

if isfield(opts,'coef')
    if opts.adaptive
        [D,Dt] = FD3D_adaptive(ko,opts.coef);
        fprintf('Using rewieghted finite difference transform\n');
    end
end

% Check that Dt is the true adjoint of D
%[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
%if ~flg
%    error('Sparse domain transforms do not appear consistent');
%end


function [] = final_disp(out,opts)

fprintf('Number of total iterations is %d. \n',out.total_iter);
fprintf('Final error: %5.3f\n',out.final_error);
fprintf('Final ||w||_1: %5.3g\n',out.final_wl1);
fprintf('Final ||Du-w||^2: %5.3g\n',out.final_Du_w);

if opts.disp
    figure(2);
    plot(out.lam1);  %plot lam1, ||W||_1
    hold on;
    plot(out.lam3.*out.mu);  %plot lam3, mu||Au -f||^2
    plot(abs(out.f));   %plot f, the objective function
    plot(out.mu)
    plot(2:opts.inner_iter:max(size(out.f)),out.f(2:opts.inner_iter:end),'kx','Linewidth',2);
    legend({'||W||_1','mu*||Au - f||_2^2',...
        'objective function','mu','multipliers updated'},...
        'Location','northeast');
    xlabel('iteration');
    title('Values in each iteration');
    hold off;

end