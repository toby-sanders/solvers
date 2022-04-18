% testing to minimize ||Ax-b||^2 subject to Cx-d>=0
d = 200; % signal dimension, x
m = 300; % number of least squares terms/rows in A
c = 100; % number of inequality constraints
rng(321);
opts.iter = 1000;
opts.tol = 1e-8;

% randomly generate A, b, C, and D
A = rand(m,d);
C = rand(c,d);
b = rand(m,1);
dv = rand(c,1);

% solve LS inequality contraint problem with Matlabs lsqlin and my solvers
tic;
xM = lsqlin(A,b,-C,-dv,[],[]);
t1 = toc;
xtest = lsqlin(A,b,[],[],C,dv);
tic;
[Mx,out] = myLSineq(A,b,C,dv,opts);
t2 = toc;
tic;
[Mx2,out2] = myLSineqExact(A,b,C,dv,opts);
t3 = toc;
% xx = linespace(-1,1,100);
% [X,Y] = meshgrid(xx,xx);

% compute inequality errors
v11 = norm(min(C*Mx - dv,0));% myrel(C*Mx,dv);
v12 = norm(min(C*Mx2 - dv,0));% myrel(C*Mx,dv);
v22 = norm(min(C*xM - dv,0));% myrel(C*xM,dv);

% compute objective function values
v1 = A*Mx-b;
v10 = A*Mx2-b;
v2 = A*xM-b;
v1 = v1'*v1;
v10 = v10'*v10;
v2 = v2'*v2;

%%
% print and display
fprintf('MATLAB, constraint error: %g, L2 norm: %g, time: %g\n',v22,v2,t1);
fprintf('MYver, constraint error: %g, L2 norm: %g, time: %g\n',v11,v1,t2);
fprintf('MYver exact, constraint error: %g, L2 norm: %g, time: %g\n',v12,v10,t3);
figure(99);
subplot(2,2,1);hold off;
plot(xM,'x');hold on;
plot(Mx,'o');
plot(Mx2,'o');
legend('matlab','me');title('values of soln');
hold off;
subplot(2,2,2);semilogy(out.rel_chg);
subplot(2,2,3);hold off;
plot(C*xM-dv,'x');hold on;
plot(C*Mx-dv,'o');
plot(out.lambda,'*');hold off;
legend('matlab','me','LM values');title('inequ cond');
subplot(2,2,4);hold off;
plot(out.lambda,'x');title('LM recovered');hold on;
plot(out2.mu,'o');
hold off;