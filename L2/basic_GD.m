function [U,out] = basic_GD(A,b,p,q,r,iter)

% algorithm for basic gradient decent for LS

% unify implementation of A
if ~isa(A,'function_handle')
    A = @(u,mode) f_handleA(A,u,mode);
end

U = zeros(p*q*r,1);
out.alpha = zeros(iter,1);
out.rel_error = zeros(iter,1);
nrmb = b'*b;
for i = 1:iter
   Aub = A(U,1)-b;
   g = A(Aub,2); 
   Ag = A(g,1);
   alpha = Ag'*Aub/max(Ag'*Ag,eps);  % steepest decent value
   U = U - alpha*g;
   out.alpha(i) = alpha;
   out.rel_error(i) = Aub'*Aub/nrmb;
end
U = reshape(U,p,q,r);