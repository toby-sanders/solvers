function [u,out] = MHOL2_Fourier(S, b, V, n, parm)
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
if isfield(parm,'theta'); theta = parm.theta;
else, theta = 1e3; end
tic
[m, ~] = size(b);
k = parm.order;
% Ab = A(b,2);

% store regularization operator eigenvalues
v = zeros(nlev,n);
for i = 1:nlev
    v(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,n-1,n)/n).^(2*k+2))./...
            (sin(pi*linspace(0,n-1,n)/n).^2)*(2^(2-k-i)/nlev)^2;
    v(i,1) = 0;
end
v = sum(v,1)';
bstr = zeros(n,1);
bstr(S) = b;
u = ones(n,1);

iter = 0;
% fixed-point iteration
out.sigmas = [];out.etas = [];out.thetas = theta; out.errors = [];
out.trHiAA = []; out.trHiV = [];
while true
  uo = u;
  u = sqrt(n)*real(ifft(bstr./(1+theta*v))); % multiply by root n so unitary
  % error = norm(u - uo, 1)/n;
  error = myrel(u,uo,2);
  out.errors = [out.errors;error];
  if error <= tol || iter == maxiter; break; end  
  iter = iter + 1;  
  trHiAA = sum(1./(theta*v(S) + 1));
  v2 = theta*v;
  v2(S) = v2(S) + 1;
  trHiV = sum(v./v2);
  r = fft(u); r = r(S)/sqrt(n); % divide by root n to make unitary
  r = b - r;
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
