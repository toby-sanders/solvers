% testing to minimize ||Ax-b||^2 subject to Cx-d>=0
clear;
d = 200; % signal dimension, x
m = 300; % number of least squares terms/rows in A
c = 50; % number of inequality constraints
rng(321);
opts.iter = 10000;
opts.tol = 1e-8;

% randomly generate A, b, C, and D
A = rand(m,d);
C = rand(c,d);
b = rand(m,1);
dv = rand(c,1);

% solve LS inequality contraint problem with Matlabs lsqlin and my solvers
try
    tic;
    xM = lsqlin(A,b,-C,-dv,[],[]);
    t1 = toc;
    xtest = lsqlin(A,b,[],[],C,dv);
    matOptBox = true;
catch
    matOptBox = false;
end
tic;
[Mx,out] = myLSineqOpers(A,b,C,dv,opts);
t2 = toc;
tic;
[Mx2,out2] = myLSineq(A,b,C,dv,opts);
t3 = toc;
% xx = linespace(-1,1,100);
% [X,Y] = meshgrid(xx,xx);

% compute inequality errors
v11 = norm(min(C*Mx - dv,0));% myrel(C*Mx,dv);
v12 = norm(min(C*Mx2 - dv,0));% myrel(C*Mx,dv);
if matOptBox
    v22 = norm(min(C*xM - dv,0));% myrel(C*xM,dv);
    v2 = A*xM-b;
    v2 = v2'*v2;
end

% compute objective function values
v1 = A*Mx-b;
v10 = A*Mx2-b;

v1 = v1'*v1;
v10 = v10'*v10;


%%
% print and display
if matOptBox
    fprintf('MATLAB, constraint error: %g, L2 norm: %g, time: %g\n',v22,v2,t1);
end
fprintf('MYver, constraint error: %g, L2 norm: %g, time: %g\n',v11,v1,t2);
fprintf('MYver exact, constraint error: %g, L2 norm: %g, time: %g\n',v12,v10,t3);
figure(99);
if matOptBox
    subplot(2,2,1);hold off;
    plot(xM,'x');hold on;
    plot(Mx,'o');
    plot(Mx2,'o');
    legend('matlab','me');title('values of soln');
    hold off;
else
    subplot(2,2,1);hold off;
    plot(Mx,'o');hold on;
    plot(Mx2,'x');hold off;
    legend('me');title('values of soln');
    hold off;
end

subplot(2,2,2);semilogy(out.rel_chg);

if matOptBox
    subplot(2,2,3);hold off;
    plot(C*xM-dv,'x');hold on;
    plot(C*Mx-dv,'o');
    plot(out.lambda,'*');hold off;
    legend('matlab','me','LM values');title('inequ cond');
else
    subplot(2,2,3);hold off;
    % plot(C*xM-dv,'x');hold on;
    plot(C*Mx-dv,'o');hold on;
    plot(out.lambda,'*');hold off;
    legend('me','LM values');title('inequ cond');
end
subplot(2,2,4);hold off;
plot(out.lambda,'x');title('LM recovered');hold on;
hold off;
