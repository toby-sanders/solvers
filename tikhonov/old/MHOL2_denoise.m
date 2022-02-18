function [u,out] = MHOL2_denoise(b, V, n, parm)
% min_u {||A u = b||_2^2/(2*sigma2) + u'*V*u/(2*eta2)}
% where u is n by 1, (sigma2, eta2) maximize the evidence.
% Here V = sqrt(mT/rT)*T'*T, T is mT by n and of rank rT.
% Mi(theta, B) approximates (A^T A + theta V)\B  (preconditioning)
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

% store regularization operator eigenvalues
v = zeros(nlev,n);
for i = 1:nlev
    v(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,n-1,n)/n).^(2*k+2))./...
            (sin(pi*linspace(0,n-1,n)/n).^2)*(2^(2-k-i)/nlev)^2;
    v(i,1) = 0;
end
v = sum(v,1)';

iter = 0;
u = b;  % initial approximation for CG
% fixed-point iteration
out.sigmas = [];
out.etas = [];
out.thetas = theta;
out.errors = [];
while true
  uo = u;
  u = real(ifft(fft(b)./(1+theta*v)));
  % error = norm(u - uo, 1)/n;
  error = myrel(u,uo,2);
  out.errors = [out.errors;error];
  %%
  if error <= tol || iter == maxiter; break; end  
  iter = iter + 1;
  trHiAA = sum((1+theta*v).^(-1));
  trHiV = 1/theta*sum(1-(1+theta*v).^(-1));
  assert(m - trHiAA > 0);
  assert(n - k - theta*trHiV > 0);
  r = b - u;
  Vu = V(u);
  sigma2 = r'*r/(m - trHiAA);
  eta2 = u'*Vu/(n - k - theta*trHiV);
  theta = sigma2/eta2;
  out.sigmas = [out.sigmas;sigma2];
  out.etas = [out.etas;eta2];
  out.thetas = [out.thetas;theta];
end
out.iter = iter;
fprintf('*** MHOL2 ***\n');
toc
fprintf('Estimated relative error is %f.\n', error/(norm(u, 1)/n));
fprintf('Number of iterations is %i.\n', iter);
fprintf('sigma = %f, eta = %f.\n', sigma2^(1/2), eta2^(1/2));
return
