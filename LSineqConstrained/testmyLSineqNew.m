% testing to minimize ||Ax-b||^2 subject to Cx-d>=0
d = 200; % signal dimension, x
m = 100; % number of least squares terms/rows in A
c = 100; % number of inequality constraints
e = 100; % number of equality contraints
rng(321);
opts.iter = 10000;
opts.tol = 1e-8;

% randomly generate A, b, C, and D
A = rand(m,d);
C = rand(c,d);
b = rand(m,1);
dv = rand(c,1);
E = rand(e,d);
ev = rand(e,1);

tic;
[x,out] = myLSineq2(A,b,E,ev,C,dv,opts);
t2 = toc;



ineqE = C*x-d;

figure(23);tiledlayout(2,2);
nexttile;
plot(ineqE);
nexttile;
plot(out.x0);
nexttile;
semilogy(out.GD.rel_chg);
