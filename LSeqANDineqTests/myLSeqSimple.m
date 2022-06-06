function [x,out] = myLSeq(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx = d
% using a standard lagrange multiplier method

% According to the first order KKT conditions, we can instead solve the
% problem
%       Bz = y, 
% where
%       B = [A^T*A  C^T ;  C   0] ,  y = [A^T*b ; d].
% This is very simple if all of these operators are small, as in this
% algorithm

% Note that B is symmetric pos. def., so we can solve this equation using a
% conjugate gradient method directly.


AtA = A'*A;
B = [AtA, C'; C zeros(size(C,1))];
z = [A'*b;d];
% x = B\z;
x0 = my_cgsSPD(B,z,500,1e-10);
x = x0(1:size(A,2));
out.lambda = x0(size(A,2)+1:end);