d = 4;
m = 30;
c = 2;
rng(321);
opts.iter = 1000;
opts.tol = 1e-6;

A = rand(m,d);
C = rand(c,d);
b = rand(m,1);
dv = rand(c,1);

tic;
xM = lsqlin(A,b,[],[],C,dv);
t1 = toc;
tic;
[Mx,out] = myLSeq(A,b,C,dv,opts);
t2 = toc;
% xx = linespace(-1,1,100);
% [X,Y] = meshgrid(xx,xx);

v11 = myrel(C*Mx,dv);
v22 = myrel(C*xM,dv);
v1 = A*Mx-b;
v2 = A*xM-b;
v1 = v1'*v1;
v2 = v2'*v2;

fprintf('MATLAB, constraint error: %g, L2 norm: %g, time: %g\n',v22,v2,t1);
fprintf('MYver, constraint error: %g, L2 norm: %g, time: %g\n',v11,v1,t2);

figure(99);
subplot(2,2,1);hold off;
plot(xM,'x');hold on;
plot(Mx,'o');
hold off;
subplot(2,2,2);semilogy(out.rel_chg);
