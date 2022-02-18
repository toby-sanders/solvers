function [U, out] = l1optimo(A,b,p,q,r,opts)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code has been modified to solve general l1 penalty problems
%Edited by: Toby Sanders, ASU

%%%
%original algorithm description


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


% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end

% get or check opts
opts = check_l1_opts(opts);
%ko = opts.order;
%wrap_shrink = opts.wrap_shrink;


% Sparse domain transforms
global D Dt
D = opts.D;
Dt = opts.Dt;
% Check that Dt is the true adjoint of D
%[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
%if ~flg
%    error('Sparse domain transforms do not appear consistent');
%end



% problem dimension
n = p*q*r;


% mark important constants
mu = opts.mu;
beta = opts.beta;
tol_inn = opts.tol_inn;
tol_out = opts.tol;
gam = opts.gam;



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

% initialize U, beta
muf = mu;       
betaf = beta;     % final beta
[U,beta,mu] = init_U(p,q,r,Atb,scl,opts);    % U: p*q
if mu > muf; mu = muf; end
if beta > betaf; beta = betaf; end
muDbeta = mu/beta;
rcdU = U;
nrmrcdU = norm(rcdU(:));
nrmb = norm(b);


% initialize multiplers
%sigma = zeros(p,q,r);                       % sigmax, sigmay: p*q
%delta = zeros(length(b),1);                % delta: m

% initialize D^T sigma + A^T delta
DtsAtd = zeros(n,1);

% initialize out.errTrue which records the true relative error
if isfield(opts,'Ut')
    Ut = opts.Ut*scl;        %true U, just for computing the error
    nrmUt = norm(Ut(:));
else
    Ut = [];
end 
if ~isempty(Ut)
    out.errTrue = norm(U(:) - Ut(:));
end


% prepare for iterations
out.res = [];      % record errors of inner iterations--norm(H-Hp)
out.reer = [];     % record relative errors of outer iterations
out.innstp = [];   % record RelChg of inner iterations
out.itrs = [];     % record # of inner iterations
out.itr = Inf;     % record # of total iterations
out.f = [];        % record values of augmented Lagrangian fnc
out.cnt = [];      % record # of back tracking
out.lam1 = []; out.lam2 = []; out.lam3 = []; out.lam4 = []; out.lam5 = [];
out.tau = []; out.alpha = []; out.C = []; out.mu = [];

gp = [];


Ucoeff = D(U);                   % Ux, Uy, Uz: p*q*z

% first shrinkage step
Wcoeff = max(abs(Ucoeff) - 1/beta, 0).*sign(Ucoeff);

% initialize multiplers
sigma = zeros(size(Ucoeff));                       % sigma
delta = zeros(length(b),1);                % delta: m
%reset edge values if not using wrap around regularization
%if ~wrap_shrink
%    if ko<=q, Wx(:,end-ko+1:end,:) = Ux(:,end-ko+1:end,:); end;
%    if ko<=p, Wy(end-ko+1:end,:,:) = Uy(end-ko+1:end,:,:); end;
%    if ko<=r, Wz(:,:,end-ko+1:end) = Uz(:,:,end-ko+1:end); end;
%end

lam1 = sum(sum(sum(abs(Wcoeff))));

[lam2,lam3,lam4,lam5,f,g2,Au,g] = get_g(U,Ucoeff,Wcoeff,...
    lam1,beta,mu,A,b,Atb,sigma,delta);
%lam, f: constant      g2: pq        Au: m         g: pq

% compute gradient
d = g2 + muDbeta*g - DtsAtd;

count = 1; sum_itrs = 0;
Q = 1; C = f;                     % Q, C: costant
out.f = [out.f; f]; out.C = [out.C; C];
out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; out.lam3 = [out.lam3; lam3];
out.lam4 = [out.lam4; lam4]; out.lam5 = [out.lam5; lam5];out.mu = [out.mu; mu];
if opts.disp
    fprintf('Beginning outer iteration #%d\n',count);
    fprintf('iter    ||w||_1    ||Du - w||^2  ||Au - b||^2\n');
end
        
for ii = 1:opts.maxit
    %if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
    %    fprintf('outer iter = %d, inner iter = %d, total iter = %d, \n'...
    %        ,count,ii-sum_itrs,ii);
    %end

    % compute tau first
    if ~isempty(gp)
        dg = g - gp;                        % dg: pq
        dg2 = g2 - g2p;                     % dg2: pq
        ss = uup'*uup;                      % ss: constant
        sy = uup'*(dg2 + muDbeta*dg);       % sy: constant
        % sy = uup'*((dg2 + g2) + muDbeta*(dg + g));
        % compute BB step length
        tau = abs(ss/max(sy,eps));               % tau: constant

        %fst_itr = false;
    else
        % do Steepest Descent at the 1st ieration
        %d = g2 + muDbeta*g - DtsAtd;         % d: pq
        dcoeff = D(reshape(d,p,q,r));                   %dx, dy: p*q*r
        %dDd = sum(sum(sum(dcoeff.^2)));         % dDd: cosntant
        dDd = sum(sum(sum(dcoeff.*conj(dcoeff))));
        Ad = A(d,1);                        %Ad: m
        % compute Steepest Descent step length
        tau = abs((d'*d)/(dDd + muDbeta*Ad'*Ad));

        % mark the first iteration
        %fst_itr = true;
    end

    % keep the previous values for backtracking
    Up = U; gp = g; g2p = g2; Aup = Au; 
    Ucoeffp = Ucoeff; DtsAtdp =  DtsAtd;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ONE-STEP GRADIENT DESCENT %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    taud = tau*d;
    U = U(:) - taud;
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
    Ucoeff = D(U);

    [lam2,lam3,lam4,lam5,f,g2,Au,g] = get_g(U,Ucoeff,Wcoeff,...
        lam1,beta,mu,A,b,Atb,sigma,delta);

    % Nonmonotone Line Search
    alpha = 1;
    du = U - Up;
    const = opts.c*beta*(d'*taud);

    % Unew = Up + alpha*(U - Up)
    cnt = 0; flag = true;
    
    while f > C - alpha*const
        if cnt == 5
            % shrink gam
            gam = opts.rate_gam*gam;

            % give up and take Steepest Descent step
            if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
                disp('    count of back tracking attains 5 ');
            end

            %d = g2p + muDbeta*gp - DtsAtd;
            dcoeff = D(reshape(d,p,q,r));
            %dDd = sum(sum(sum(dx.^2 + dy.^2 + dz.^2)));
            dDd = sum(sum(sum(dcoeff.*conj(dcoeff))));
            Ad = A(d,1);
            tau = abs((d'*d)/(dDd + muDbeta*Ad'*Ad));
            U = Up(:) - tau*d;
            % projected gradient method for nonnegtivity
            if opts.nonneg
                U = max(real(U),0);
            end
            U = reshape(U,p,q,r);
            Ucoeff = D(U);
            Ubar = Ucoeff - sigma/beta;
            Wcoeff = max(abs(Ubar) - 1/beta, 0).*sign(Ubar);
            %reset edge values if not using wrap around regularization
            %if ~wrap_shrink
            %    if ko<=q, Wx(:,end-ko+1:end,:) = Ux(:,end-ko+1:end,:); end;
            %    if ko<=p, Wy(end-ko+1:end,:,:) = Uy(end-ko+1:end,:,:); end;
            %    if ko<=r, Wz(:,:,end-ko+1:end) = Uz(:,:,end-ko+1:end); end;
            %end

            lam1 = sum(sum(sum(abs(Wcoeff))));
            [lam2,lam3,lam4,lam5,f,g2,Au,g] = get_g(U,Ucoeff,Wcoeff,...
                lam1,beta,mu,A,b,Atb,sigma,delta);
            alpha = 0; % remark the failure of back tracking
            break;
        end
        if flag
            dg = g - gp;
            dg2 = g2 - g2p;
            dAu = Au - Aup;% dAu: m
            dUcoeff = Ucoeff - Ucoeffp;
            flag = false;
        end
        alpha = alpha*opts.gamma;
        [U,lam2,lam3,lam4,lam5,f,Ucoeff,Au,g,g2] = update_g(p,q,r,...
            lam1,alpha,beta,mu,Up,du,gp,dg,g2p,dg2,Aup,dAu,Wcoeff,...
            Ucoeffp,dUcoeff,b,sigma,delta);
            %Uxp,dUx,Uyp,dUy,Uzp,dUz,b,sigma,delta);
        cnt = cnt + 1;
    end
    

    % if back tracking is successful, then recompute
    if alpha ~= 0
        Ubar = Ucoeff - sigma/beta;
        Wcoeff = max(abs(Ubar) - 1/beta, 0).*sign(Ubar);
        %reset edge values if not using wrap around regularization
        %if ~wrap_shrink
        %    if ko<=q, Wx(:,end-ko+1:end,:) = Ux(:,end-ko+1:end,:); end;
        %    if ko<=p, Wy(end-ko+1:end,:,:) = Uy(end-ko+1:end,:,:); end;
        %    if ko<=r, Wz(:,:,end-ko+1:end) = Uz(:,:,end-ko+1:end); end;
        %end


        % update parameters related to Wx, Wy
        [lam1,lam2,lam4,f,g2] = update_W(beta,...
            Wcoeff,Ucoeff,sigma,lam1,lam2,lam4,f);
    end

    % update reference value
    Qp = Q; Q = gam*Qp + 1; C = (gam*Qp*C + f)/Q;
    uup = U - Up; uup = uup(:);           % uup: pq
    nrmuup = norm(uup);                   % nrmuup: constant

    out.res = [out.res; nrmuup];
    out.f = [out.f; f]; out.C = [out.C; C]; out.cnt = [out.cnt;cnt];
    out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; out.lam3 = [out.lam3; lam3];
    out.lam4 = [out.lam4; lam4]; out.lam5 = [out.lam5; lam5];
    out.tau = [out.tau; tau]; out.alpha = [out.alpha; alpha];out.mu = [out.mu; mu];

    if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
       % fprintf('  ||W||_%5.3f,   ||D(U)-W|| = %5.3f, ||Au-f||/||f|| = %5.3f, ',...
        %    lam1 ,sqrt(lam2), sqrt(lam3)/nrmb);
        %fprintf('||W||_1=%5.3f, ||Au-f||^2 = %5.3f, ',...
        %    lam1, lam3);
        prnt_format = '%3.0f %10.5g %12.5g %13.5g\n';
        fprintf(prnt_format, ii-sum_itrs,lam1,lam2,lam3);
    end

    if ~isempty(Ut)
        errT = norm(U(:) - Ut(:));
        out.errTrue = [out.errTrue; errT];
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            fprintf('  ||Utrue-U||(/||Utrue||) = %5.3f(%5.3f%%), ',errT, 100*errT/nrmUt);
        end
    end

    % recompute gradient
    d = g2 + muDbeta*g - DtsAtd;

    % compute relative change or optimality gap
    if opts.StpCr == 1          % relative change
        nrmup = norm(Up(:));
        RelChg = nrmuup/nrmup;
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            fprintf('    ||Uprvs-U||/||Uprvs|| = %5.3f; \n', RelChg);
        end
    else                       % optimality gap
        RelChg = norm(d);
        %if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
        %    fprintf('    optimality gap = %5.3f; \n', RelChg);
        %end
    end

    out.innstp = [out.innstp; RelChg];

    if (RelChg < tol_inn || ii-sum_itrs >= opts.maxin)% && ~fst_itr
        count = count + 1;
        RelChgOut = norm(U(:)-rcdU(:))/nrmrcdU;
        out.reer = [out.reer; RelChgOut];
        rcdU = U;
        nrmrcdU = norm(rcdU(:));
        if isempty(out.itrs)
            out.itrs = ii;
        else
            out.itrs = [out.itrs; ii - sum(out.itrs)];
        end
        sum_itrs = sum(out.itrs);

        % stop if already reached final multipliers
        if RelChgOut < tol_out || count > opts.maxout || sqrt(lam3(end))/nrmb<opts.min_l2_error
            if sqrt(lam3(end))/nrmb<opts.min_l2_error
                fprintf('\nREACHED OPTIMAL L2 ERROR!!!\n\n');
            end
            if opts.isreal
                U = real(U);
            end
            if exist('scl','var')
                U = U/scl;
            end
            out.itr = ii;
            out.final_error = norm(A(scl*U(:),1)-b)/nrmb;
            fprintf('Number of total iterations is %d. \n',out.itr);
            fprintf('Final error: %5.3f\n',out.final_error);
            if opts.disp
                figure(2);
                plot(out.lam1);  %plot lam1, ||W||_1
                hold on;
                plot(out.lam3.*out.mu);  %plot lam3, mu||Au -f||^2
                plot(abs(out.f));   %plot f, the objective function
                plot(out.mu)
                plot(2:opts.maxin:ii,abs(out.f(2:opts.maxin:end)),'kx','Linewidth',2);
                legend({'||W||_1','mu*||Au - f||_2^2',...
                    'objective function','mu','multipliers updated'},...
                    'Location','Southeast');
                xlabel('iteration');
                title('Values in each iteration');
                hold off;
                %figure(1);
                %imagesc(U(:,:,round(r/2)));colormap(gray);colorbar;
                
            end
            return
        end

        % update multipliers
        [sigma,delta,lam4,lam5,f] = update_mlp(beta,mu, ...
            Wcoeff,Ucoeff,Au,b,sigma,delta,lam4,lam5,f);
        
        
        % update penality parameters for continuation scheme
        beta0 = beta;
        beta = min(betaf, beta*opts.rate_ctn);
        mu = min(muf, mu*opts.rate_ctn);
        %beta = beta*opts.rate_ctn;
        %mu = mu*opts.rate_ctn;
        %if beta > betaf; beta = betaf; end
        %if mu > muf; mu = muf; end
        muDbeta = mu/beta;

        % update function value, gradient, and relavent constant
        f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;
        DtsAtd = -(beta0/beta)*d;     % DtsAtd should be divded by new beta instead of the old one for consistency!  
        d = g2 + muDbeta*g - DtsAtd;

        %initialize the constants
        gp = [];
        gam = opts.gam; Q = 1; C = f;
        
        
        if opts.disp
            fprintf('**************************************');
            fprintf('\n    LAGRANGIAN MULTIPLIERS UPDATED \n');
            fprintf('    Beginning outer iteration #%d\n',count);
            fprintf('    mu = %d , beta = %d\n',mu,beta);
            fprintf('iter    ||w||_1    ||Du - w||^2  ||Au - b||^2\n');
        end
    end

end

if opts.isreal
    U = real(U);
end
if exist('scl','var')
    fprintf('Attain the maximum of iterations %d. \n',opts.maxit);
    U = U/scl;
end




function [lam2,lam3,lam4,lam5,f,g2,Au,g] = get_g(U,Ucoeff,Wcoeff,...
    lam1,beta,mu,A,b,Atb,sigma,delta)
global Dt

% A*u
Au = A(U(:),1);

% g
g = A(Au,2) - Atb;



% lam2
Vcoeff = Ucoeff-Wcoeff;
lam2 = sum(sum(sum(Vcoeff.*conj(Vcoeff))));

% g2 = D'(Du-w)
g2 = Dt(Vcoeff);


% lam3
Aub = Au-b;
lam3 = norm(Aub)^2;

%lam4
lam4 = sum(sum(sum(sigma.*Vcoeff)));

%lam5
lam5 = delta'*Aub;

% f
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;



function [U,lam2,lam3,lam4,lam5,f,Ucoeff,Au,g,g2] = update_g(p,q,r,lam1,...
    alpha,beta,mu,Up,du,gp,dg,g2p,dg2,Aup,dAu,Wcoeff,...
    Ucoeffp,dUcoeff,b,sigma,delta)
    %Uxp,dUx,Uyp,dUy,Uzp,dUz,b,sigmax,sigmay,sigmaz,delta)

g = gp + alpha*dg;
g2 = g2p + alpha*dg2;
U = Up + alpha*reshape(du,p,q,r);
Au = Aup + alpha*dAu;
Ucoeff = Ucoeffp + alpha*dUcoeff;

Vcoeff = Ucoeff - Wcoeff;

lam2 = sum(sum(sum(Vcoeff.*conj(Vcoeff))));
Aub = Au-b;
lam3 = norm(Aub)^2;
lam4 = sum(sum(sum(sigma.*Vcoeff)));
lam5 = delta'*Aub;
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4 - lam5;



function [lam1,lam2,lam4,f,g2] = update_W(beta,...
    Wcoeff,Ucoeff,sigma,lam1,lam2,lam4,f)
global Dt

% update parameters because Wx, Wy were updated
tmpf = f -lam1 - beta/2*lam2 + lam4;
lam1 = sum(sum(sum(abs(Wcoeff))));
Vcoeff = Ucoeff-Wcoeff;
g2 = Dt(Vcoeff);
lam2 = sum(sum(sum(Vcoeff.*conj(Vcoeff))));
lam4 = sum(sum(sum(sigma.*Vcoeff)));
f = tmpf +lam1 + beta/2*lam2 - lam4;



function [sigma,delta,lam4,lam5,f] = update_mlp(beta,mu, ...
    Wcoeff,Ucoeff,Au,b,sigma,delta,lam4,lam5,f)

Vcoeff = Ucoeff - Wcoeff;
sigma = sigma - beta*Vcoeff;
Aub = Au-b;
delta = delta - mu*Aub;

tmpf = f + lam4 + lam5;
lam4 = sum(sum(sum(sigma.*Vcoeff)));
lam5 = delta'*Aub;
f = tmpf - lam4 - lam5;





function [U,beta,mu] = init_U(p,q,r,Atb,scl,opts)

% initialize beta
if isfield(opts,'beta0') && isfield(opts,'mu0')
    beta = opts.beta0;
    mu = opts.mu0;
else
    error('Initial mu or beta is not provided.');
end

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
