% this script file is for testing my equality contrained least squares
% solvers. More specifically, it solves the following problem:
%   min_x || Ax-b ||_2  s.t. Cx=d


clear;
d = 400; % size of x
m = 200; % # of equations in linear system, A
c = 100; % # of equality contraints
rng(321);
opts.iter = 5000; % max iterations
opts.tol = 1e-9; % convergence tolerance

% generate random matrix and vectors
A = rand(m,d);
C = rand(c,d);
b = rand(m,1);
dv = rand(c,1);

% if we have the optimization toolbox, compare with matlab
try
    tic;
    xM = lsqlin(A,b,[],[],C,dv);
    t1 = toc;
    matOptBox = false;
catch
    matOptBox = false;
end

% run the solvers
tic;
[Mx,out0] = myLSeq(A,b,C,dv,opts);
t2 = toc;
[Mx2,out] = myLSeqGrad(A,b,C,dv,opts);

% some metrics
v11 = myrel(C*Mx,dv);
v1 = A*Mx-b;
v1 = v1'*v1;

%% compare results
if matOptBox
    v22 = myrel(C*xM,dv);
    v2 = A*xM-b;
    v2 = v2'*v2;
    fprintf('MATLAB, constraint error: %g, L2 norm: %g, time: %g\n',v22,v2,t1);
end
fprintf('MYver, constraint error: %g, L2 norm: %g, time: %g\n',v11,v1,t2);

if matOptBox
    figure(99);
    subplot(2,2,1);hold off;
    plot(xM,'x');hold on;
    plot(Mx,'o');
    hold off;
else
    figure(99);
    subplot(2,2,1);hold off;
    plot(Mx,'o');hold on;
    plot(Mx2,'x');
    hold off;
    legend('CG solver''gradient solver')
    title('recovered values')
end
subplot(2,2,2);semilogy(out.rel_chg);title('convergence of gradient solver')
subplot(2,2,3);hold off;
semilogy(abs(C*Mx-dv));hold on;
semilogy(abs(C*Mx2-dv));
hold off;
title('zero equality contraint values');
legend('CG solver''gradient solver')
