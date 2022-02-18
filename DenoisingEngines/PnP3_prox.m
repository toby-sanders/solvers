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

% if isempty(R), R = getBM3D(5/255); end
if isempty(R), R = getTNRD; end

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


Up = U;
ii = 0;  % main loop
while ii < opts.iter % iterations refers to outer loops
    ii = ii + 1;
    tau = 1;
    
    alpha = (ii-1)/(ii+2);
    Z = U + alpha*(U-Up);
    
    g = reshape((A(A(Z(:),1),2)-Atb),p,q,r);
    V = Z - tau*g;
    Up = U;
    U = R(real(V),opts.sigma0*sqrt(tau));

    rel_chg = myrel(U,Up);
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<tol, break; end
end
out.elapsed_time = toc;
out.rel_UV = myrel(U,V);


