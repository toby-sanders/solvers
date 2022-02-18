function [u,out] = HOTVL2_Fourier(S, b, n, parm)
% min_u {||A u = b||_2^2/(2*sigma2) + u'*V*u/(2*eta2)}
% where u is n by 1, (sigma2, eta2) maximize the evidence.
% Here V = sqrt(mT/rT)*T'*T, T is mT by n and of rank rT.
% Mi(theta, B) approximates (A^T A + theta V)\B  (preconditioning)

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
tic;
if numel(n)<3, n(end+1:3) = 1;
elseif numel(n)>3, error('n can have at most 3 dimensions');end
p = n(1); q = n(2); r = n(3); k = parm.order;
[R,Rt] = get_D_Dt(k,p,q,r,parm,b);
T = @(X)Rt(col(R(X)));
[m, ~] = size(b);

% store eigenvalues of regularization operator 
vx = zeros(nlev,q); vy = zeros(nlev,p); vz = zeros(nlev,r);
for i = 1:nlev
    vx(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,q-1,q)/q).^(2*k+2))./...
            (sin(pi*linspace(0,q-1,q)/q).^2)*(2^(2-k-i)/nlev)^2;
    vy(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,p-1,p)/p).^(2*k+2))./...
        (sin(pi*linspace(0,p-1,p)/p).^2)*(2^(2-k-i)/nlev)^2;
    vz(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,r-1,r)/r).^(2*k+2))./...
        (sin(pi*linspace(0,r-1,r)/r).^2)*(2^(2-k-i)/nlev)^2;
end
vx(:,1) = 0; vy(:,1) = 0; vz(:,1) = 0;
vx = sum(vx,1); vy = sum(vy,1); vz = sum(vz,1); 
V = vy' + vx + reshape(vz,1,1,r);
V2 = zeros(p,q,r); V2(S) = 1;
% stretch b
bstr = zeros(prod(n),1);
bstr(S) = b; bstr = reshape(bstr,p,q,r);
u = ones(prod(n),1);

iter = 0;
% fixed-point iteration
out.sigmas = [];out.etas = [];out.thetas = theta; out.errors = [];
out.trHiAA = []; out.trHiV = [];
while true
  uo = u;
  % multiply by root n so unitary
  u = sqrt(prod(n))*col(real(ifftn(bstr./(V2+theta*V)))); 
  errors = myrel(u,uo,2);
  out.errors = [out.errors;errors];
  if errors <= tol || iter == maxiter; break; end  
  iter = iter + 1;  
  trHiAA = sum(1./(theta*col(V(S)) + 1));
  % v2 = theta*V;
  % v2(S) = v2(S) + 1;
  trHiV = sum(col(V./(V2+theta*V)));
  % divide by root n to make unitary
  re = fftn(u); re = re(S)/sqrt(prod(n)); 
  re = b - re;
  Tu = T(u);
  sigma2 = real(re'*re/(m - trHiAA));
  eta2 = real(u'*Tu/(prod(n) - theta*trHiV));
  theta = sigma2/eta2;  
  out.sigmas = [out.sigmas;sigma2];out.etas = [out.etas;eta2];
  out.thetas = [out.thetas;theta];  
  out.trHiAA = [out.trHiAA;trHiAA];out.trHiV = [out.trHiV;trHiV]; 
end
out.iter = iter; out.elapsed_time = toc;
fprintf('*** MHOL2 ***\n');
fprintf('Estimated relative error is %f.\n', errors/(norm(u, 1)/prod(n)));
fprintf('Number of iterations is %i.\n', iter);
fprintf('sigma = %f, eta = %f.\n', sigma2^(1/2), eta2^(1/2));
u = reshape(u,p,q,r);
