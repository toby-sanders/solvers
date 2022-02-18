function X = mySolve(A, Mi, B, X)
  % solve A X = B where B may have more than one column
  % Mi(B) approx A\B
  [n, ~] = size(B);
  CGtol = 1e-4;
  CGmaxiter = 200;
  % preconditioned conjugate gradient method from Wikipedia
  AX = A(X);
  normAi = sqrt( max(sum(X.*X)./sum(AX.*AX)) ); % estimate of norm(A^-1)
  R = B - AX;
  Z = Mi(R);
  ZR = sum(Z.*R);
  P = Z;
  iter = 1;
  error = 2*CGtol;
  while error > CGtol && iter <= CGmaxiter
    AP = A(P);
    normAi = max(normAi, sqrt( max(sum(P.*P)./sum(AP.*AP)) ));
    alpha = sum(R.*Z)./sum(P.*AP);
    X = X + alpha.*P;
    R = R - alpha.*AP;
    error = normAi*sqrt( max(sum(R.*R)) );
    if error <= CGtol; return; end
    Z = Mi(R);
    ZRo = ZR; ZR = sum(Z.*R);
    beta = ZR./ZRo;
    P = Z + beta.*P;
    iter = iter + 1;
  end
%  fprintf('CG error est = %f.\n', error);
% estimate of norm(A^-1) not very good:
% fprintf('normAi = %f; norm(inv(A(eye(n)))) = %f.\n', ...
%          normAi, norm(inv(A (eye(n)) )) );

