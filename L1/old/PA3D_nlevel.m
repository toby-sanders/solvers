function [f,N, out] = PA3D_nlevel(A,b,p,q,r,opts)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modifications by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016


% This code has been modified to solve l1 penalty problems with the
% polynomial annihilation transform


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
% TVAL3 solves the corresponding augmented Lagrangian function:
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



% get and check opts
opts = check_PA_opts(opts);


% mark important constants
%tol_inn = opts.tol_inn;
%tol_out = opts.tol_out;
k = opts.order;
n = p*q*r;
wrap_shrink = opts.wrap_shrink;
if round(k)~=k
    wrap_shrink = true;
end



% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end


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


% initialize everything else
global D Dt
[mu,beta,muf,betaf,sigma,delta,ind,nrmb,epsb,out] ...
    = get_init(p,q,r,b,opts,k,wrap_shrink);    % U: p*q



% calculate A'*b
f = A(b,2);
f = reshape(f,p,q,r);
fc = D(f);

% first shrinkage step
w = max(abs(fc) - 1/beta, 0).*sign(fc);
% reset edge values if not using periodic regularization
if ~wrap_shrink, w(ind)=fc(ind); end

%lam1 = sum(sum(sum(abs(w))));
%N = zeros(size(b));
xi = opts.xi;
gD=Dt(D(f)-w);
Af = A(f(:),1);
N = (b-Af)/5;
%N(:)=0;
%gAf = A(Af-b+N,2);
Afnb = Af+N - b;


for ii = 1:opts.outer_iter
    gsigma = Dt(sigma);
    gdelta = A(delta,2);
    
    if opts.disp
            fprintf('    Beginning outer iteration #%d\n',ii);
            fprintf('    mu = %d , beta = %d , order = %g\n',mu,beta,k);
            fprintf('iter    ||w||_1    ||Du - w||^2  ||Au - b||^2\n');
    end
        
    for jj = 1:opts.inner_iter
        c = (ii-1)*opts.inner_iter + jj;
        % f subproblem
        gAf = A(Afnb,2);
        
        gf = mu*gAf + beta*gD - gsigma - gdelta; %gradient of f subproblem
        
        % compute step length
        if jj==1
            Dgf = D(gf);
            Agf = A(gf,1);
            %tauf = s234/mu*s5;
            s1 = gf'*gf;
            s2 = mu*(Agf')*Agf;
            s3 = beta*(col(Dgf)')*col(Dgf);
            tauf = -s1/(s2+s3);
        else
            % BB-like step length
            % not as accurate
            % quicker to compute
            dgAf = gAf - gAfp;
            dgD = gD - gDp;
            s1 = ffp'*ffp;
            s2 = ffp'*(beta*dgD + mu*dgAf);
            tauf = -s1/s2;
        end
        
        fp = f; gAfp = gAf; gDp = gD;  % keep previous values
        
        
        f = f(:) + tauf*gf(:);  % gradient decent
        if opts.nonneg
            f = max(f,0);  % project function into subspace
        end
        
        
        f = reshape(f,p,q,r);
        fc = D(f);
        ffp = f-fp;ffp = ffp(:);
        
        
        
        
        % shrinkage
        fcbar = fc - sigma/beta;
        w = max(abs(fcbar) - 1/beta, 0).*sign(fcbar);
        % reset edge values if not using periodic regularization
        if ~wrap_shrink, w(ind)=fc(ind); end
        
        Af = A(f(:),1);
        Afnb = Af+N-b;
        %gAf = A(Afnb,2);% update gradient A
        v = fc-w;
        gD = Dt(v); % update gradient D
        
        % update function values
        lam1 = sum(sum(abs(w)));
        lam2 = v(:)'*v(:);
        lam3 = Afnb'*Afnb;
        lam4 = sigma(:)'*v(:);
        lam5 = delta'*(Afnb);
        
        objf = lam1+beta/2*lam2 + mu/2*lam3 - lam4 - lam5;
       
        
        
        
        % n subproblem
        if norm(Af-b)/nrmb <opts.error+.05
            
            nN = N'*N;
            gN = mu*Afnb + xi*n.*(1 - epsb/sqrt(nN))- delta;  % gradient of n subproblem
            % compute taun
            s1 = gN'*(delta - mu*Afnb - N);
            ngN = gN'*gN;

            NgN = N'*gN;

            taun = 0;
            for tt = 1:10
               taun = (s1+xi*eps*NgN/(nN + taun^2*ngN + 2*taun*NgN)^(0.5))...
                   /((mu + xi)*ngN - eps*xi*ngN/(nN + taun^2*ngN + 2*taun*NgN)^(0.5));
            end

            %taun = -1/4/(mu+beta);

            Np = N;
            N = N + taun*gN;  %gradient decent
            Afnb = Afnb + N - Np; % update Afnb
            %norm(N)
            %norm(gN)
        end
        
        
        
        
        out.lam1(c) = lam1; out.lam2(c) = lam2;
        out.lam3(c) = lam3; out.lam4(c) = lam4;
        out.lam5(c) = lam5;
        out.objf(c) = objf;
        
        
        if (opts.disp > 0) && (mod(ii,opts.disp) == 0)
            prnt_format = '%3.0f %10.5g %12.5g %13.5g\n';
            fprintf(prnt_format, jj,lam1,lam2,lam3);
        end
        
    end
    
    % update penality parameters for continuation scheme
    %beta0 = beta;
    beta = min(betaf, beta*opts.rate_ctn);
    mu = min(muf, mu*opts.rate_ctn);
    
    % update multipliers
    [sigma,delta,~,~] = update_mlp(beta,mu,...
        w,fc,Afnb,sigma,delta);
   
    
end

out.total_iter = jj*ii;
out.final_error = norm(Af+N-b)/nrmb;
out.final_wl1 = lam1(end);
out.final_Du_w = lam2(end);
final_disp(out,opts);
% rescale U
f = f/scl;
N = N/scl;
%N = reshape(N,p,q,r);
norm(A(f(:),1)-b)/norm(b)
norm(b)
norm(N)












function [sigma,delta,lam4,lam5] = update_mlp(beta,mu, ...
    w,fc,Afnb,sigma,delta)


v = fc - w;
sigma = sigma - beta*v;
delta = delta - mu*(Afnb);


%tmpf = f + lam4 + lam5;
lam4 = sum(sum(sum(sigma.*v)));
lam5 = delta'*(Afnb);
%f = tmpf - lam4 - lam5;







function [mu,beta,muf,betaf,sigma,...
    delta,ind,nrmb,epsb,out] = get_init(p,q,r,b,opts,k,wrap_shrink)

% initialize U, beta, mu
muf = opts.mu;       % final mu
betaf = opts.beta;     % final beta
beta = opts.beta0;
mu = opts.mu0;

nrmb = norm(b);
epsb = (opts.error*nrmb);






% declare out variables
nn = opts.outer_iter*opts.inner_iter;
out.reer = zeros(nn,1);     % record relative errors of outer iterations
out.obj = zeros(nn,1);        % record values of augmented Lagrangian fnc
out.cnt = zeros(nn,1);      % record # of back tracking
out.lam1 = zeros(nn,1); out.lam2 = zeros(nn,1); out.lam3 = zeros(nn,1); 
out.lam4 = zeros(nn,1); out.lam5 = zeros(nn,1);
out.tau = zeros(nn,1); out.alpha = zeros(nn,1); out.C = zeros(nn,1); 
out.mu = zeros(nn,1);



global D Dt
if opts.smooth_phase
    if round(k) == k
        [D,Dt] = FD3D(k,p,q,r);
    else
        [D,Dt] = FFD3D(k,p,q,r); 
    end
else
    if sum(sum(sum(abs(opts.phase_angles))))==0
        opts.phase_angles = exp(-1i*reshape(Atb,p,q,r));
    else
        opts.phase_angles = exp(-1i*opts.phase_angles);
    end
    if round(k) == k
        [D,Dt] = FD3D_complex(k,p,q,r,opts.phase_angles);
    else
        [D,Dt] = FFD3D_complex(k,p,q,r,opts.phase_angles);
    end
end

if isfield(opts,'coef')
    if opts.adaptive
        [D,Dt] = FD3D_adaptive(k,opts.coef);
        fprintf('Using rewieghted finite difference transform\n');
    end
end

% Check that Dt is the true adjoint of D
[flg,~,~] = check_D_Dt(D,Dt,[p,q,r]);
if ~flg
    error('Sparse domain transforms do not appear consistent');
end

% initialize multiplers
sigma = D(zeros(p,q,r));                       
delta = zeros(size(b,1),1);


if ~wrap_shrink
    ind = get_ind(k,p,q,r);
else
    ind=[];
end



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
    plot(abs(out.objf));   %plot f, the objective function
    plot(out.mu)
    plot(2:opts.inner_iter:max(size(out.objf)),out.objf(2:opts.inner_iter:end),'kx','Linewidth',2);
    legend({'||W||_1','mu*||Au - f||_2^2',...
        'objective function','mu','multipliers updated'},...
        'Location','northeast');
    xlabel('iteration');
    title('Values in each iteration');
    hold off;

end