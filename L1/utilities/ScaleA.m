function [A,b,scl] = ScaleA(n,A,b)

% Scales mu, A and f so that the largest eigenvalue of A'*A is 1 and the
% new problem
%
% min sum_i (||wi|| + beta/2 ||Diu - wi||^2) + mu/2 ||Au - b||^2
%
% is equivalent to the old one.  
%
% If option is assigned, mu will be scaled accordingly.
%
% Written by: Chengbo Li


% check if A*A'=I
tmp = rand(numel(b),1);
if norm(A(A(tmp,2),1)-tmp,1)/norm(tmp,1) < 1e-3
    scl=1;
    return;
end

eopts.disp = 0;
eopts.tol = .05;
if ~isreal(A(rand(n,1),1))
    eopts.isreal = false;
end

fh = @(x) A(A(x,1),2);
% s2 = eigs(fh,n,1,'lm',eopts);
[tau,out] = getStepLength(A,n);
s2 = 1/tau;
if real(s2) > 1 + 1e-10
    b = b/sqrt(s2);
    A = @(x,mode) A(x,mode)/sqrt(s2);
end
scl = sqrt(s2);

return