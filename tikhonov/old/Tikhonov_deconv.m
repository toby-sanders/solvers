function [U,out] = Tikhonov_deconv(h,b,n,opts)

% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_u    mu*||h*u-b||^2 + ||Du||^2
% using the exact formula by Fourier filter
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 01/29/2018

% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = prod(n);

opts = check_tik_opts(opts);

uhat = fft(b);
hhat = fft(h);
k = opts.order;

v = 2^(1-k)*(exp(-1i*2*pi/p*linspace(0,p-1,p)')-1).^k;
v = v.*conj(v);
if q==p, v = col(v + v'); end
U = reshape(ifft(((hhat.*conj(hhat) + v/opts.mu).^(-1))...
    .*(hhat.*uhat)),p,q,r);
out.hhat = hhat;






