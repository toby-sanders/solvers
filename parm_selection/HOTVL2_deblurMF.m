function [u,out] = HOTVL2_deblurMF(h,b, parm)
% Fast deconvolution with Tikhonov regularization and automated parameter
% selection.  This algorithm solves:
% min_u {||h*u - b||_2^2/(2*sigma2) + ||Tu||_2^2/(2*eta2)}
% where (sigma2, eta2) maximize the evidence (likelihood of the data).
% The approach is described in the article
% "Maximum Evidence Algorithms for Automated Parameter Selection in 
% Regularized Inverse Problems," by Sanders, Platte, and Skeel

% written by Toby Sanders @Lickenbrock Tech.
% last update: 1/17/2020

% optional parameters
% tic;
if isfield(parm, 'levels'); nlev = parm.levels;
else, nlev = 1;parm.levels = 1; end
if isfield(parm, 'tol'); tol = parm.tol;
else, tol = 0.0001; end
if isfield(parm, 'iter'); iter = parm.iter;
else, iter = 50; end
if isfield(parm,'theta'); theta = parm.theta;
else, theta = 1e3; end

[p,q,nF] = size(b); k = parm.order;
Fb = fft2(b);
% [R,Rt] = get_D_Dt(k,p,q,r,parm);
% T = @(X)Rt(col(R(X)));
m = numel(b);

% store eigenvalues of regularization operator 
vx = zeros(nlev,q); vy = zeros(nlev,p); % vz = zeros(nlev,r);
for i = 1:nlev
    vx(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,q-1,q)/q).^(2*k+2))./...
            (sin(pi*linspace(0,q-1,q)/q).^2)*(2^(2-k-i)/nlev)^2;
    vy(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,p-1,p)/p).^(2*k+2))./...
        (sin(pi*linspace(0,p-1,p)/p).^2)*(2^(2-k-i)/nlev)^2;
    % vz(i,:) = 4^k*(sin(2^(i-1)*pi*linspace(0,r-1,r)/r).^(2*k+2))./...
     %    (sin(pi*linspace(0,r-1,r)/r).^2)*(2^(2-k-i)/nlev)^2;
end
vx(:,1) = 0; vy(:,1) = 0; % vz(:,1) = 0;
vx = sum(vx,1); vy = sum(vy,1); % vz = sum(vz,1); 
V = vy' + vx;%  + reshape(vz,1,1,r);

hhat = fft2(h,p,q);
hhat2 = sum(hhat.*conj(hhat),3);

cnt = 0;
u = sum(b,3);
Fu = fft2(u);
out.sigmas = [];out.etas = [];
out.thetas = theta;out.errors = [];
out.trHiV = [];out.trHiAA = [];
numer = sum(Fb.*conj(hhat),3);
while true
  % new solution and check convergence
  Fuo = Fu;
  Fu = numer./(hhat2+theta*V);
  errors = norm(Fu(:) - Fuo(:),2)/norm(Fuo(:),2);
  % errors = myrel(Fu,Fuo);
  out.errors = [out.errors;errors];
  if errors <= tol || cnt == iter; break; end  
  cnt = cnt + 1;
  
  % exact traces
  trHiAA = sum(hhat2(:)./(hhat2(:)+theta*V(:)));
  trHiV = sum(V(:)./(hhat2(:)+theta*V(:)));
  % trHiV = 1/theta*(prod(n)-trHiAA);

  % update parameters
  % re = b - ifftn(Fu.*hhat);
  re = (Fb - Fu.*hhat)/sqrt(p*q);
  FuV = Fu.*V/(p*q);
  % Tu = T(u);
  sigma2 = sum((real(re(:).*conj(re(:)))))/(m - trHiAA);
  eta2 = real(sum(Fu(:).*conj(FuV(:)))/(p*q - theta*trHiV));
  % eta2 = u(:)'*Tu(:)/(p*q*r - theta*trHiV);
  theta = sigma2/eta2;
  
  % output variables
  out.sigmas = [out.sigmas;sigma2];
  out.etas = [out.etas;eta2]; out.thetas = [out.thetas;theta];
  out.trHiAA = [out.trHiAA;trHiAA]; out.trHiV = [out.trHiV;trHiV];
end
u = real(ifft2(Fu));
% fprintf('HOTVL2, order %i, %i level(s)\n',k,parm.levels);
% fprintf('-----------------------------\n');
% out.elapsed_time = toc;
% fprintf('Estimated relative error is %f.\n', errors/(norm(u, 1)/(p*q*r)));
% fprintf('Number of iterations is %i.\n',cnt);
% fprintf('sigma = %f, eta = %f.\n', sigma2^(1/2), eta2^(1/2));
