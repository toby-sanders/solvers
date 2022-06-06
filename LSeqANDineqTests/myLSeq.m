function [x,out] = myLSeq(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx = d
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
if ~isa(C,'function_handle'), C = @(u,mode) f_handleA(C,u,mode); end

Atb = A(b,2);
dimx = numel(Atb);



z = [Atb;d]; % right hand side vector

% operator that performs evaluation of B matrix
B = @(x)myLMeqOperators(A,C,x,dimx);

% run CG and then separate desired vector and lagrange multiplier
x0 = my_cgsSPD(B,z,500,1e-10);
x = x0(1:dimx);
out.lambda = x0(dimx+1:end);


% this function operator performs evaluation of B matrix
function y = myLMeqOperators(A,C,x,dim1)

y1 = x(1:dim1);
y2 = x(dim1+1:end);
y3 = A(A(y1,1),2); %AtA*x
y4 = C(y2,2); %Ct times multiplier
y5 = C(y1,1); % C times x
y = cat(1,y3+y4,y5); % concatonate terms

