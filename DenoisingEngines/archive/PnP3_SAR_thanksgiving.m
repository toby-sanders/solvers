function [U, out] = PnP3(A,R,b,n,opts)


% Written by Toby Sanders @Lickenbrock Tech.
% 10/21/2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plug and Play image reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [U, out] = PnP3(A,R,b,n,beta,opts)
%
% Motivation is to find:
%
%               min_f { mu/2*||Au - b||_2^2 + R(u) }
%
% where R is an input denoising operator, e.g. if V is a noisy image, the
% R(V) is the denoised image
% Generally speaking, this formulation does not makes sense, since R is not
% a functional, however, the method derived from the ADMM approach in an
% approach known as "plug-and-play" (PnP)

% R can be input as empty, in which case the BM3D denoiser is used

% Inputs: 
%   A: matrix operator as either a matrix or function handle
%   R: an image denoising operator
%   b: data values in vector form
%   n: image/ signal dimensions in vector format
%   opts: structure containing input parameters, 
%       see function check_HOTV_opts.m for these
% Outputs:
%   U: reconstructed signal
%   out: output numerics

tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
opts = check_PnP_opts(opts);  % get and check opts

if isempty(R), R = getBM3D(5); end
% if isempty(R), R = getTNRD; end

% mark important constants
tol = opts.tol; 
tol_inn = max(tol,1e-4);  % inner loop tolerance isn't as important
n = p*q*r;

% check that A* is true adjoint of A
% check scaling of parameters, maximum constraint value, etc.
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end; clear flg;
[~,~,scl] = ScaleA(n,A,b);
% [~,scl] = Scaleb(b);
% if opts.scale_mu, opts.mu = opts.mu*scl; end
% opts.beta = opts.beta*scl;

% initialize everything else
Atb = A(b,2); % A'*b
if size(opts.init,1)==p, U = opts.init;
else, U = reshape(Atb/scl^2,p,q,r);
end
V = U; % splitting variable, V
mu = opts.mu/scl^2; beta = opts.beta;
sigma = zeros(p,q,r); % Lagrange multiplier
out.rel_chg_out = [];rel_chg_out = 1;
out.rel_chg_inn = [];


gA = mu*(A(A(U(:),1),2)-Atb);
gSplit = beta*(U-V) + sigma;
ii = 0;  % main loop
while ii < opts.iter % iterations refers to outer loops
    ii = ii + 1;
    g = gA + gSplit(:);
    % optimal step length at the 1st iteration
    Ag = A(g,1);
    denomObj = mu*(Ag'*Ag) + beta*(g'*g); 
    tau = abs((g'*g)/denomObj);
    % check for convergence after multiplier updates
    if ii~=1
        rel_chg_out = norm(tau*g)/norm(U(:));
        out.rel_chg_out = [out.rel_chg_out; rel_chg_out];
        if rel_chg_out < tol, break; end
    end
    
    for jj = 1:opts.inner_iter
        % compute step length, tau
        if jj~=1
            % BB-step length            
            ss = uup'*uup;                      
            sy = uup'*(g-gp);       
            tau = abs(ss/max(sy,eps));   
        end
        % descent
        Up = U; 
        U = U - tau*reshape(g,p,q,r); 
       
        % projected gradient method for inequality constraints
        if opts.nonneg, U = max(real(U),0);
        elseif opts.isreal, U = real(U); end
        uup = U(:) - Up(:);
  
        rel_chg_inn = norm(uup)/norm(Up(:));
        
        % recompute gradient
        gp = g;
        gA = mu*(A(A(U(:),1),2)-Atb);
        gSplit = beta*(U-V) + sigma;
        g = gA + gSplit(:);
           
        % check inner loop convergence
        out.rel_chg_inn = [out.rel_chg_inn; rel_chg_inn];
        if (rel_chg_inn < tol_inn) || ii>= opts.iter, break; end 
    end
    
    % denoising step and multiplier update
    theta = angle(U);% +sigma/beta);
    V = R(abs(U + sigma/beta));
    V = V.*exp(1i*theta);
    sigma = sigma + beta*(U-V);
    % display for prototyping
    if opts.disp
    fprintf('multiplier update number %i, relChgOut = %g\n',ii,rel_chg_out);
    figure(345);
    subplot(2,2,1);imagesc(abs(U));title('u');colorbar;
    subplot(2,2,2);imagesc(abs(V));title('v');colorbar;
    subplot(2,2,3);imagesc(abs(sigma));title('sigma');colorbar;
    subplot(2,2,4);semilogy(out.rel_chg_out);title('rel change out');
    end
    % recompute gradient
    gA = mu*(A(A(U(:),1),2)-Atb);
    gSplit = beta*(U-V) + sigma;
end
out.elapsed_time = toc;
out.rel_UV = myrel(U,V);


