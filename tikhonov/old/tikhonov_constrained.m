function [X,out] = tikhonov_constrained(A,b,dim1,dim2,dim3,opts)
%This algorithm uses conjugate gradient method to find
%
%  X = argmin_x ||Dx||_2^2  s.t. Ax = b
%
%Here D is a finite difference operator who's order can be specified in the
%opts structure
%The fields for the opts structure are defined in get_tikhonov_local


% Written by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 02/09/2017







[A,Amain,bpad,b,tol,maxit,x0,scl,opts,delta,datadim] ...
    = get_tikhonov_local(A,b,dim1,dim2,dim3,opts);

X = A(bpad,2);
nrmb = norm(b);

for i = 1:maxit
    Axb = A(X,1)-bpad;
    out.err(i) = norm(Axb(1:datadim))/nrmb;
    
    g = A(Axb,2);
    if mod(i,10) ==0
        delta = delta - Axb;
        delta(datadim+1:end) = 0;
        %delta(:) = 0;
        g = g - A(delta,2);
    end
    g2 = A(g,1);
    sd = 2*opts.lambda*Axb'*g2 - g2'*delta;
    sn = 2*opts.lambda*sum(col(g2.*conj(g2)));
    alpha = sd/max(sn,eps);
    out.alpha(i) = alpha;
    out.relchg(i) = alpha*norm(g)/norm(X);
    X = X - alpha*g;
    fprintf('iter = %i, error = %g, relchg = %g\n',...
    i,out.err(i),out.relchg(i));

end

X = reshape(X,dim1,dim2,dim3)/scl;

    
    
    
    





function [A,Amain,bpad,b,tol,maxit,x0,scl,opts,delta,datadim] = get_tikhonov_local(A,b,dim1,dim2,dim3,opts)

%order of the finite difference operator
if ~isfield(opts,'order')
    opts.order = 1;
end

%lambda is the penalty weight for the second norm
if ~isfield(opts,'lambda')
    opts.lambda = .15;%/sqrt(factorial(2*(opts.order))/factorial(opts.order)^2);
    fprintf('lambda not specified, set to default\n');
    fprintf('Default lambda = %f\n',opts.lambda);
end


%scale lambda according to the order
if ~isfield(opts,'scale_lam')
    opts.scale_lam = false;
end

if opts.scale_lam
    opts.lambda = opts.lambda/...
        sqrt(factorial(2*(opts.order))/factorial(opts.order)^2);
end

%maximum iterations for iterative method
if ~isfield(opts,'maxit')
    opts.maxit=50;
end
maxit = opts.maxit;

%tolerance for convergence
if ~isfield(opts,'tol')
    opts.tol = 1e-3;
end
tol = opts.tol;

%initial guess
if ~isfield(opts,'x0')
    opts.x0 = zeros(dim1,dim2,dim3);
end
x0 = opts.x0(:);



%scaling of the operator A and vector b, scaling usually suggested so that
%consistent values for mu and beta may be used independent of the problem
if isfield(opts,'scale_A')
    if ~islogical(opts.scale_A)
        error('opts.scale_A should be true or false.');
    end
else
    opts.scale_A = true;
end


if isfield(opts,'scale_b')
    if ~islogical(opts.scale_b)
        error('opts.scale_b should be true or false.');
    end
else
    opts.scale_b = true;
end

Amain = A;
if ~isa(Amain,'function_handle')
    Amain = @(u,mode) f_handleA(Amain,u,mode);
end

% check scaling A
if opts.scale_A
    [Amain,b] = ScaleA(dim1*dim2*dim3,Amain,b);
end

% check scaling b
scl = 1;
if opts.scale_b
    [b,scl] = Scaleb(b);
end
datadim = size(b,1);


%pad b for the regularization term
bpad = [b;zeros(3*dim1*dim2*dim3,1)];

%define the finite difference operators
[D,Dt] = FD3D(opts.order,dim1,dim2,dim3);

%build the operator [A;D] and [A^t + Dt]
A = @(u,mode)tikhonov_operator_local(Amain,D,Dt,dim1,dim2,dim3,datadim,u,mode,opts.lambda);

% initialize lagrange multiplier
delta = zeros(numel(bpad),1);














function y = tikhonov_operator_local(A,D,Dt,dim1,dim2,dim3,datadim,u,mode,lambda)


%A is a function handle that represents the forward operator
%D is the finite difference transform
%Dt is the transpose of D
%dim variables are dimension of the signal
%datadim is the number of data points
%u is the input signal 
%mode is 1 or 2
%lambda is the penalty constant
switch mode
    case 1
        y1 = A(u(:),1);
        y2 = D(reshape(u,dim1,dim2,dim3));
        y2 = lambda*[y2(:)];
        y=[y1;y2];
        
    case 2
        y1 = A(u(1:datadim),2);
        dd = dim1*dim2*dim3;
        x0 = u(datadim+1:datadim+dd);
        y0 = u(datadim+dd+1:datadim+2*dd);
        z0 = u(datadim+2*dd+1:end);
        y2 = Dt([x0,y0,z0]);
        y = y1+lambda*y2;
        
end
