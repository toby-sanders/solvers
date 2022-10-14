function [x,out] = Tikhonov(A,b,n,opts)


% These functions solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% subject to optional inequality constaints
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.

% Fields in the opts structure (defaults are assigned for empty fields):
% order - order of the finite difference reg. operator, D
% iter - maximum number of iterations for CG
% mu - regularization parameter (see formulation above)
% tol - convergence tolerance for CG
% levels - default is 1, but for higher integers it uses a multiscale
% operators for D

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/25/2018

opts = check_tik_opts(opts);

if opts.nonneg 
    % warning('implementation of nonnegativity constraint is slower');
    [x,out] = Tikhonov_Nesta(A,b,n,opts);
else
    [x,out] = Tikhonov_CG(A,b,n,opts);
end
% fprintf('completed in %g seconds\n',out.total_time);
% fprintf('iters = %i, ||Ax-b||/||b|| = %g\n',out.iters,out.rel_error(end));