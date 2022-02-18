function [x] = Tikhonov_CG(A,b,n,opts)

% This function solves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_x    mu*||Ax-b||^2 + ||Dx||^2
% using a CG decent method
% D is a finite difference operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options are more or less the same as HOTV3D, see check_hotv_opts or the
% users guide.

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 01/29/2018

% set image dimensions
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions'); end
p = n(1); q = n(2); r = n(3);
n = p*q*r;

opts = check_tik_opts(opts);

% unify implementation of A, check scaling A
if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
if opts.scale_A, [A,b] = ScaleA(p*q*r,A,b); end

[D,Dt] = get_D_Dt(opts.order,p,q,r,opts,A(b,2));
datadim = size(b,1);
%pad b for the regularization term
if opts.order ~= 0, bpad = [sqrt(opts.mu)*b;zeros(opts.levels*3*p*q*r,1)];
else, bpad = [sqrt(opts.mu)*b;zeros(p*q*r,1)]; end
% special case for 2D MHOTV
if opts.levels>1 && r==1
    bpad = [sqrt(opts.mu)*b;zeros(opts.levels*2*p*q*r,1)];
end

Amain = A;

%build the operator [A;D] and [A^t + Dt]
A = @(u,mode)tikhonov_operator_local(Amain,D,Dt,...
    p,q,r,datadim,u,mode,sqrt(opts.mu),opts.order,opts.levels);
[flg,~,~] = check_D_Dt(@(u)A(u,1),@(u)A(u,2),[n,1]);
if ~flg, error('A and A* do not appear consistent'); end
clear flg;

[x,out] = my_cgs(A,bpad,opts.iter,opts.tol);
% B = @(x)A(A(x,1),2);
% y = A(bpad,2);
% x = cgs(B,y,1e-5,opts.iter);
x = reshape(x,p,q,r);

function y = tikhonov_operator_local(A,D,Dt,p,q,r,datadim,u,mode,sqmu,order,levels)
% A is a function handle that represents the forward operator
% D is the finite difference transform
% Dt is the transpose of D
% p,q,r variables are dimension of the signal
% datadim is the number of data points
% u is the input signal 
% mu is the penalty constant
switch mode
    case 1
        y = sqmu*A(u(:),1);
        y2 = D(reshape(u,p,q,r));
        y=[y;y2(:)];
    case 2
        y = A(u(1:datadim),2);
        dd = p*q*r*levels;
        if order ~=0
            x0 = u(datadim+1:datadim+dd);
            y0 = u(datadim+dd+1:datadim+2*dd);
            % special case for 2D MHOTV
            if r~=1 || levels==1, z0 = u(datadim+2*dd+1:end);
            else, z0 = []; end
            y0 = Dt([x0,y0,z0]);
        else
            y0 = u(datadim+1:end);
        end
        
        y = sqmu*y+y0;
        
end
        
