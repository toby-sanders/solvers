% simple test for Bob's fixed point scheme for SURE
clear;
example = 'boxcar'; order = 1; levels = 1;
% example = 'hat'; order = 2; levels = 2;
% example = 'sine'; order = 3; levels = 3;
% example = 'pwquad'; order = 2; levels = 3;
n = 100; % signal dimension
m = n;  % number of samples
SNR = 3;   % snr
tol = 1e-5; % tolerance for convergence of solution 
% (based on parameter changes)
lambda0 = 1e2; % initial parameter selection
maxiter = 50;
levels = 1;

% signal
[A,x,b,sigma] = my_1D_examples(SNR,n,m,example);
D = zeros(n);
D(1) = 1;
D(2:n+1:end) = - 1;
D(n+2:n+1:end) = 1;
D = D^order; % order k differences
T = D'*D;
AtA = A'*A;
Atb = A'*b;

nQ = 6;
Q = randn(m,nQ);
Q = A'*Q;


lambda = lambda0;
lambdas = zeros(maxiter,1);
estimators = zeros(maxiter,1);
trueer = zeros(maxiter,1);
estccr = zeros(maxiter,1);

estimators2 = zeros(maxiter,1);
trueer2 = zeros(maxiter,1);
estccr2 = zeros(maxiter,1);
trace_true = zeros(maxiter,1);
trace_est = zeros(maxiter,1);
for i = 1:maxiter
    % compute H and solution
    H = AtA+lambda*T; 
    Hi = inv(H);
    u = Hi*Atb;
    
    % residuals and estimate errors
    re = A*u-b;
    trace_true(i) = trace(Hi*T*Hi*AtA);
    estimators(i) = -m*sigma^2 + re'*re + 2*sigma^2*trace(Hi*AtA);
    estccr(i) = max(estimators(i),eps); % just a correction for the plot
    trueer(i) = norm(A*(u-x),2)^2;
    
    % redo everything with numerical estimates
    u2 = cgs(H,Atb,1e-15,300);
    % trapp = zeros(n,nQ);
    for j = 1:nQ
        trapp = cgs(H,Q(:,j),1e-10,300);
        V(:,j) = D*trapp;
    end        
    trace_est(i) = sum(col(V.*V))/nQ;
    
    % residuals and estimate errors
    re2 = A*u2-b;
    estimators2(i) = -m*sigma^2 + re2'*re2 + 2*sigma^2*trace(Hi*AtA);
    estccr2(i) = max(estimators2(i),eps); % just a correction for the plot
    trueer2(i) = norm(A*(u2-x),2)^2;
    
    
    % update for lambda based on Bob's fixed point scheme
    v1 = sigma^2*trace_est(i);
    v2 = T*u;
    v3 = v2'*Hi*v2;
    lambda = v1/v3;
    lambdas(i) = lambda;
end

opts.order = order;
opts.levels = levels;
opts.maxiter = maxiter;
opts.mu = 1/lambda0;
[u3,out] = UPRE_fixedpt(A,b,[n,1,1],opts,sigma);
%%
figure(45);
subplot(2,2,1);semilogy(estccr);hold on;
semilogy(out.UPREs*m);
semilogy(trueer); xlabel('iters');
legend({'SURE estimator','SURE est app','true error'});hold off;
subplot(2,2,2);semilogy(lambdas);hold on;
semilogy(1./out.mus); legend('rec','rec app');
title('lambdas');xlabel('iters');hold off;
subplot(2,2,3);plot(u);hold on;
plot(u3);
plot(x);legend('recovered','recovered app','true');hold off;
subplot(2,2,4);
plot(trace_true);hold on;
plot(trace_est); hold off; 
legend('true traces','est traces');xlabel('iters');




