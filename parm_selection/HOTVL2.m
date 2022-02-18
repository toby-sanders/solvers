function [u,out] = HOTVL2(A, b, Mi, n, parm)
% min_u {||A u = b||_2^2/(2*sigma2) + u'*V*u/(2*eta2)}
% where u is n by 1, (sigma2, eta2) maximize the evidence.
% Here V = sqrt(mT/rT)*T'*T, T is mT by n and of rank rT.
% Mi(theta, B) approximates (A^T A + theta V)\B  (preconditioning)
if ~isa(A,'function_handle'); A = @(u,mode) f_handleA(A,u,mode); end
% required parameter: parm.order
% optional parameters:
if ~isfield(parm, 'levels'); parm.levels = 1; end
if isfield(parm, 'tol'); tol = parm.tol;
else, tol = 1e-4; end
if isfield(parm, 'maxiter'); maxiter = parm.maxiter;
else, maxiter = 50; end
if isfield(parm, 'ntrace'); nQ = parm.ntrace;
else, nQ = 6; end
if isfield(parm,'theta'); theta = parm.theta;
else, theta = 1e3; end
tic;

if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3);
[R,Rt] = get_D_Dt(parm.order,p,q,r,parm,A(b,2));
V = @(X)Rt(col(R(X)));
m = numel(b); k = parm.order;
H = @(theta) @(X) A(A(X,1),2) + theta*V(X);
Ab = A(b,2);
% probing vectors for approximate traces
[Q, ~] = qr(randn(prod(n), nQ), 0);
% AAQ = A(A(Q, 1), 2);  % A'*A*Q
AAQ = zeros(prod(n),nQ);
VQ = zeros(prod(n),nQ);
for i = 1:nQ
    VQ(:,i) = V(Q(:,i)); 
    AAQ(:,i) = A(A(Q(:,i),1),2);
end
iter = 0;
u = Ab;  % initial approximation for CG
HiQ = Q;  % initial approximation for CG
% fixed-point iteration
out.sigmas = [];out.etas = [];
out.thetas = theta;out.errors = [];
out.trHiV = [];out.trHiAA = [];
while true
  uo = u;
  Htheta = H(theta);
  Mitheta = @(B) Mi(theta, B);
  u = mySolve(Htheta, Mitheta, Ab, u);
  % errors = norm(u - uo, 1)/prod(n);
  errors = myrel(u,uo);
  out.errors = [out.errors;errors];
  if errors <= tol || iter == maxiter; break; end  
  iter = iter + 1;
  
  % approximate the traces
  for ii = 1:nQ
    HiQ(:,ii) = mySolve(Htheta, Mitheta, Q(:,ii), HiQ(:,ii));  % H\Q
  end
  trHiAA = prod(n)/nQ*trace(HiQ'*AAQ);  % tr(H\A'*A)
  trHiV = prod(n)/nQ*trace(HiQ'*VQ);  % tr(H\V)
  crrctn_fctr = prod(n)/(trHiAA + theta*trHiV);
  trHiAA = crrctn_fctr*trHiAA;  
  assert(m - trHiAA > 0);
  trHiV = crrctn_fctr*trHiV;  
  assert(prod(n) - theta*trHiV > 0);
  
  % update sigma, eta, theta
  re = b - A(u,1); Vu = V(u);
  sigma2 = re'*re/(m - trHiAA);
  eta2 = u'*Vu/(prod(n) - theta*trHiV);
  theta = sigma2/eta2;
  out.sigmas = [out.sigmas;sigma2]; out.etas = [out.etas;eta2];
  out.thetas = [out.thetas;theta];
  out.trHiAA = [out.trHiAA;trHiAA]; out.trHiV = [out.trHiV;trHiV];
end
fprintf('HOTVL2, order %i, %i level(s)\n',k,parm.levels);
fprintf('-----------------------------\n');
out.elapsed_time = toc;
fprintf('Estimated relative error is %f.\n', errors/(norm(u, 1)/prod(n)));
fprintf('Number of iterations is %i.\n', iter);
fprintf('sigma = %f, eta = %f.\n', sigma2^(1/2), eta2^(1/2));
u = reshape(u,p,q,r);
