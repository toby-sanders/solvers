function [U, out] = PnP3(A,b,n,opts)


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

sZ = 32; % size of blocks
LTopts.blockSize = sZ;
LTopts.numMax = 16; % max number of blocks to match for each ref. block
LTopts.numMin = 16; % min number of blocks to match for each ref. block
LTopts.wname = 'bior';
LTopts.wnamez = 'db';
LTopts.order = 13;
LTopts.orderz = 1;
LTopts.levels = 3;
LTopts.cycleSpin = 2;
LTopts.matchSpins =  [0 4 sZ/2-1];
LTopts.wiener = false;
LTopts.tauMode = 2;
LTopts.filtType = 'median';
sigma = .02;
R = @(z)LTBM3D(z,sigma,LTopts);


% mark important constants
tol = opts.tol; 
n = p*q*r;

% check that A* is true adjoint of A
% check scaling of parameters, maximum constraint value, etc.
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end; clear flg;

% rescale so that gradient step length is just 1
[A,b,~] = ScaleA(n,A,b); 


% initialize everything else
Atb = A(b,2); % A'*b
if size(opts.init,1)==p, U = opts.init;
else, U = reshape(Atb,p,q,r);
end
out.rel_chg = [];

tau = 1;
innerIter = 1;

Up = U;
ii = 0;  % main loop
while ii < opts.iter % iterations refers to outer loops
    ii = ii + 1;
    
    
    % alpha = (ii-1)/(ii+2);
    for j = 1:innerIter
        alpha = 0;
        Z = U + alpha*(U-Up);

        g = reshape((A(A(Z(:),1),2)-Atb),p,q,r);
        U = Z - tau*g;
        rel_chg = myrel(U,Up);
        out.rel_chg = [out.rel_chg;rel_chg];
        Up = U;
    end
    Up = U;
    theta = angle(U);
    U = R(abs(U)).*exp(1i*theta);
    % U = localDenoise(V,R,opts.sigma0,tau);

    rel_chg = myrel(U,Up);
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<tol, break; end
end
out.elapsed_time = toc;
% out.rel_UV = myrel(U,V);


function U = localDenoise(V,R,sigma0,tau)

gam = .5;
theta = angle(V);
% logU = log(abs(V));
U = abs(V).^gam;
% logU = abs(U+sigma/beta);
mm1 = min(U(:));
mm2 = max(U(:));
U = (U-mm1)/(mm2-mm1);
U = R(U,sigma0*sqrt(tau));

U = U*(mm2 - mm1) + mm1;
U = (U.^(1/gam)).*exp(1i*theta);
% U = exp(U).*exp(1i*theta);
% U = R(V);

