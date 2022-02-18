% simple test for Bob's fixed point scheme for SURE
clear;
rng(4343);
% example = 'boxcar'; order = 3; levels = 1;
% example = 'hat'; order = 2; levels = 2;
% example = 'sine'; order = 3; levels = 3;
example = 'pwquad'; order = 2; levels = 3;
n = 500; % signal dimension
m = n;  % number of samples
SNR = 3;   % snr
tol = 1e-4; % tolerance for convergence of solution 
% (based on parameter changes)
lambda0 = 1e2; % initial parameter selection
maxiter = 30;
% levels = 1;

% signal
[A,x,b,sigma] = my_1D_examples(SNR,n,m,example);
% A = randn(m,n);
% x = randn(n,1);
% b = A*x;
% b = b + randn(m,1)*mean(abs(b))/SNR;
% sigma = mean(abs(b))/SNR;

opts.order = order;
opts.levels = levels;
opts.maxiter = maxiter;
opts.tol = tol;opts.mu = 1/lambda0;
opts.theta = lambda0;
opts.scale_A = false;
[u3,out] = UPRE_sigma(A,b,[n,1,1],opts,sigma);
[u2,out2] = UPRE(A,b,[n,1,1],opts,sigma);
dAA = sum(A.*A)';  % diagonal of A'*A
Mi = @(theta, B) (dAA + theta).\B;
[u4,out4] = HOTVL2(A,b,Mi,[n,1,1],opts);
%%
fprintf('upre error = %g\n',myrel(u3,x));
fprintf('ME error = %g\n',myrel(u4,x));
fprintf('upre lambda = %g\n',1/out.mus(end));
fprintf('ME lambda = %g\n',out4.thetas(end));

figure(45);
subplot(2,2,1);hold off;
plot(x);hold on;
plot(u3);
plot(u4);hold off;
legend('true','upre','ME');
subplot(2,2,2);hold off;
semilogy(1./out.mus);hold on;
semilogy(out4.thetas);hold off;
legend('upre','ME');title('lambdas');
subplot(2,2,3);hold off;
semilogx(1./out2.mus(1:end-1),out2.UPREs);title('upre estimates');
xlabel('lambda');ylabel('UPRE estimate');
subplot(2,2,4);plot(u2);title('upre global search');


