% testing to minimize ||Ax-b||^2 subject to Cx-d>=0
clear;
d = 200; % signal dimension, x
m = 500; % number of least squares terms/rows in A
c = 50; % number of inequality constraints
e = 50; % number of equality contraints
rng(321);
opts.iter = 20000;
opts.tol = 1e-8;

% randomly generate A, b, C, and D
A = rand(m,d);
C = rand(c,d);
b = rand(m,1);
dv = rand(c,1);
E = rand(e,d);
ev = rand(e,1);

opts.gam = 1e-2;
opts.LMupdate = 2;
[x,out] = myLSineqFull(A,b,E,ev,C,dv,opts);
[x2,out2] = myLSineq(A,b,C,dv,opts);

eq1 = E*x-ev;
eq2 = E*x2-ev;

ineq1 = C*x-dv;
ineq2 = C*x2-dv;
%%
ineqE = C*x-dv;
ineqE2 = C*x2-dv;
figure(23);tiledlayout(2,2);
nexttile;hold off;
plot(ineqE,'o');hold on;
plot(ineqE2,'x');
legend('full solver','inequality only solver')
title('recovered values');
hold off;
nexttile;hold off;
semilogy(out.rel_chg);hold on;
semilogy(out2.rel_chg);hold off;
legend('full solver','inequality only solver')
nexttile;hold off;
semilogy(abs(eq1));hold on;
semilogy(abs(eq2),':r');hold off;
title('equality contraint values');
legend('full solver','inequality only solver')
nexttile;hold off;
plot((ineq1));hold on;
plot((ineq2),':r');
title('inequality contraint values');
legend('full solver','inequality only solver')
