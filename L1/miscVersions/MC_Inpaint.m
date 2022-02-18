function X = MC_Inpaint(M,S,iter,tol)

tau = 5;
delta = 1;


[m,n] = size(M);
Sc = 1:m*n;
Sc(S) = '';
Y = zeros(m,n);
X = zeros(m,n);
for i = 1:iter
    Xp = X;
    [U,Sig,V] = svd(Y,'econ');
    
    Sig = max(Sig-tau,0);
    X = U*Sig*V';
    
    Z = M-X;
    Z(Sc) = 0;
    Y = Y + delta*Z;
%     figure(90);
%     subplot(2,2,1);
%     imagesc(X);title('X');
%     subplot(2,2,2);imagesc(Y);title(num2str(i));
%     subplot(2,2,3);imagesc(Sig);
%     pause;
    ee = myrel(X,Xp);
    if ee<tol, break; end
end