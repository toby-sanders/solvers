function [x,out] = myLSeq(A,b,E,e,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Ex = e and Cx-d>=0
% using a standard lagrange multiplier method

% According to the first order KKT conditions, we can instead solve the
% problem
%       Bz = y, 
% where
%       B = [A^T*A  C^T ;  C   0] ,  y = [A^T*b ; d].
% Note that B is symmetric pos. def., so we can solve this equation using a
% conjugate gradient method directly.

% This version uses operator forms for all of the matrices, so it should in
% priciple still work for large scale problems/operators

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
if ~isa(E,'function_handle'), E = @(u,mode) f_handleA(E,u,mode); end
if ~isa(C,'function_handle'), C = @(u,mode) f_handleA(C,u,mode); end

Atb = A(b,2);
dimx = numel(Atb);
dime = numel(e);
dimd = numel(d);


z = [Atb;e]; % right hand side vector

% operator that performs evaluation of B matrix
B = @(x)myLMeqOperators(A,E,x,dimx);

% run CG and then separate desired vector and lagrange multiplier
x0 = my_cgsSPD(B,z,500,1e-10);
out.x0 = x0(1:dimx);
out.lambda = x0(dimx+1:end);


inEqus = C(out.x0,1)-d;
activeIneqs = inEqus<0;

B = @(x)myLMineqOperators(A,E,C,x,dimx,dime,dimd,activeIneqs);
z = [Atb;e;d(activeIneqs)];


[x2,out.GD] = basic_GD_local(B,z,500,dimx+dime);
x = x2(1:dimx);
out.lambda = x0(dimx+1:dimx+dime);
out.mu = x0(dimx+dime+1:end);




% this function operator performs evaluation of B matrix
function y = myLMeqOperators(A,E,x,dim1)

y1 = x(1:dim1);
y2 = x(dim1+1:end);
y3 = A(A(y1,1),2); %AtA*x
y4 = E(y2,2); %Ct times multiplier
y5 = E(y1,1); % C times x
y = cat(1,y3+y4,y5); % concatonate terms



function y = myLMineqOperators(A,E,C,x,dim1,dim2,dim3,indE)

y1 = x(1:dim1);
y2 = x(dim1+1:dim1+dim2);
y3 = A(A(y1,1),2); %AtA*x
y4 = E(y2,2); %Ct times multiplier
y5 = E(y1,1); % C times x

y0 = x(dim1+dim2+1:end);
y00 = zeros(dim3,1);
y00(indE) = y0;
y6 = C(y00,2);
y7 = C(y1,1);
y7 = y7(indE);


y = cat(1,y3+y4+y6,y5,y7); % concatonate terms




function [x,out] = basic_GD_local(A,b,iter,dim0)

x = zeros(numel(b),1);
out.rel_chg = zeros(iter,1);
% algorithm for basic gradient decent for LS
for i = 1:iter
    r = b - A(x);
    gam = r'*r/(r'*A(r));
    x = x + gam*r/2;
    x(dim0+1:end) = min(x(dim0+1:end),0);
    out.rel_chg(i) = sum(abs(gam*r(:)))/sum(abs(x(:)));
end

