function [u,out] = MHOL2(A, b, V, Mi, n, parm)
% min_u {||A u = b||_2^2/(2*sigma2) + u'*V*u/(2*eta2)}
% where u is n by 1, (sigma2, eta2) maximize the evidence.
% Here V = sqrt(mT/rT)*T'*T, T is mT by n and of rank rT.
% Mi(theta, B) approximates (A^T A + theta V)\B  (preconditioning)
if ~isa(A,'function_handle'); A = @(u,mode) f_handleA(A,u,mode); end
if ~isa(V,'function_handle'); V = @(u) f_handleA(V, u, 1); end
% required parameter: parm.order
% optional parameters:
if isfield(parm, 'levels'); nlev = parm.levels;
else, nlev = 1; end
if isfield(parm, 'tol'); tol = parm.tol;
else, tol = 1e-4; end
if isfield(parm, 'maxiter'); maxiter = parm.maxiter;
else, maxiter = 50; end
if isfield(parm, 'ntrace'); nQ = parm.ntrace;
else, nQ = 6; end
if isfield(parm,'theta'); theta = parm.theta;
else, theta = 1e3; end
tic
[m, ~] = size(b);
k = parm.order;
assert(nQ <= min(n-k, m))
H = @(theta) @(X) A(A(X,1),2) + theta*V(X);
Ab = A(b,2);
% probing vectors for approximate traces
[Q, ~] = qr(randn(n, nQ), 0);
AAQ = A(A(Q, 1), 2);  % A'*A*Q
VQ = V(Q);
iter = 0;
u = Ab;  % initial approximation for CG
HiQ = Q;  % initial approximation for CG

% fixed-point iteration
out.sigmas = [];
out.etas = [];
out.thetas = theta;
out.errors = [];
out.trHiAA = [];
out.trHiV = [];
while true
  uo = u;
  Htheta = H(theta);
  Mitheta = @(B) Mi(theta, B);
  u = mySolve(Htheta, Mitheta, Ab, u);
  % error = norm(u - uo, 1)/n;
  error = myrel(u,uo,2);
  out.errors = [out.errors;error];
  %%
  if error <= tol || iter == maxiter; break; end  
  iter = iter + 1;
  % approximate the traces
  HiQ = mySolve(Htheta, Mitheta, Q, HiQ);  % H\Q
  trHiAA = n/nQ*trace(HiQ'*AAQ);  % tr(H\A'*A)
  trHiV = n/nQ*trace(HiQ'*VQ);  % tr(H\V)
  % tr(H\A'*A) + theta*tr(H\V) == n
  crrctn_fctr = n/(trHiAA + theta*trHiV);
  trHiAA = real(crrctn_fctr*trHiAA);  
  assert(m - trHiAA > 0);
  trHiV = real(crrctn_fctr*trHiV);  
  assert(n - k - theta*trHiV > 0);
  r = b - A(u,1);
  Vu = V(u);
  sigma2 = real(r'*r/(m - trHiAA));
  eta2 = real(u'*Vu/(n - k - theta*trHiV));
  theta = sigma2/eta2;
  out.sigmas = [out.sigmas;sigma2];
  out.etas = [out.etas;eta2];
  out.thetas = [out.thetas;theta];
  out.trHiAA = [out.trHiAA;trHiAA];
  out.trHiV = [out.trHiV;trHiV];
end
out.iter = iter;
fprintf('*** MHOL2 ***\n');
toc
fprintf('Estimated relative error is %f.\n', error/(norm(u, 1)/n));
fprintf('Number of iterations is %i.\n', iter);
fprintf('sigma = %f, eta = %f.\n', sigma2^(1/2), eta2^(1/2));
return
