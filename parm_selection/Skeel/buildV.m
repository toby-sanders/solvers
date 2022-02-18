function [V, dV] = buildV(n, order, nlev)
  k = order;
  % build n by n matrix V where V = (mT/rT)*T'*T
  % where T is mT by n and of rank rT
  if (k + 1)*2^(nlev-1) - 1 > n
    fprintf('(k + 1)*2^(nlev-1) - 1 must not exceed n.\n');
    return
  end
  Phi = spdiags(ones(n,1),[0],n,n);
  for j = 1:k;
    Phi = Phi(2:end,:) - Phi(1:end-1,:);
  end
  Wi = zeros(n-k);
  mT = 0;
  for lev = 1:nlev
    t = 2^(lev-1); tm1 = t - 1;
    % Phi_t = (E^t - 1)^k (1 + E + ... + E^tm1)
    %     = (E - 1)^k (1 + E + ... + E^tm1)^(k+1)
    psi = ones(1,t);
    for j = 1:k
      nz = 1;
      for i = 1:lev-1
	psi = padarray(psi,[0 nz],'post') + padarray(psi,[0 nz],'pre');
	nz = 2*nz;
      end
    end
    ns = tm1*(k+1);  % number of superdiagonals
    Psi = spdiags(ones(n-k-ns,1)*psi, 0:ns, n-k-ns, n-k);
    Psi = 2^(1-k)/(nlev*t)*Psi;
    Wi = Wi + Psi'*Psi;
    mT = mT + n - k - ns;
  end
  % V = (mT/(n - k))*Phi'*Wi*Phi;
  V = Phi'*Wi*Phi;
  dV = diag(V);
