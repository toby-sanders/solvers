function [x,out] = myLSineqExact(A,b,C,d,opts)

% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Cx - d >= 0
% using a standard lagrange multiplier method

% this version takes advantage of the simplicity of this inequality
% contrained quadratic minimization problem. The LM, mu in this case, is
% found to be the non-positive solution to an alternative quadratic
% minimization, and the solution x is computed exactly from this

% initialize some new variables in the dual transformed problem space
Q = A'*A;
c = -A'*b;
iQ = inv(Q);
P = C*iQ*C';
z = d + C*(iQ*c);
mu = zeros(size(d,1),1);
% mu = min(P\z,0);
mu0 =  cgs(P,z);
mu = mu0;

M = size(P,1);
D = 1./(P(1:M+1:end))';
P2 = -P;P2(1:M+1:end) = 0;
% alpha = 1;
% D = P + alpha*eye(size(P,1));
% D = inv(D);
figure(172);
subplot(2,2,1);imagesc(diag(D)*P);colorbar;
subplot(2,2,2);imagesc(P);colorbar;
subplot(2,2,3);imagesc(C);colorbar;
subplot(2,2,4);imagesc(A);colorbar;


% step length tau
tau = eigs(diag(D)*P,1); tau = 1/tau;

% loop to solve for LM, mu
out.rel_chg = [];
y = mu;
for ii = 1:opts.iter
    mup = mu;
    
    mu = min(D.*(P2*mu + z),0);
    
    
    out.rel_chg = [out.rel_chg;myrel(mu,mup)];
    if out.rel_chg(end)<opts.tol, break; end
end


% exact formula for x given mu
out.mu = mu;
x = -iQ*(c + C'*mu);
