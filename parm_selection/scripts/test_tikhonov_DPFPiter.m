% a simple demo for using Tikhonov regularization
clear;
d = 500; % signal dimension
m = 100; % number of samples
SNR = 10;    % SNR, with mean zero i.i.d. Gaussian noise
order = 2; % order of the regularization

% generate a simple test signal
x = sin(3*pi*linspace(0,1,d)');
% x = zeros(d,1);x(d/4:3*d/4) = 1;
A = randn(m,d); % random sampling matrix

% generate samples and add noise
b = A*x;
sigma = mean(abs(b))/SNR;
b = b + randn(m,1)*sigma;

% algorithm parameters
tik.mu = 1e2; % regularization balancing parameter
tik.order = order;  % order of the finite differencing scheme
tik.iter = 700; % maximum number of iterations
tik.tol = 1e-16; % convergence tolerance

% compute solutions
[rec2,out2] = Tikhonov_DP_FPiter(A,b,[d,1,1],sigma,tik);
[rec3,out3] = Tikhonov_DP_FPiter_CG(A,b,[d,1,1],sigma,tik);
tik.iter = 100;
tik.maxiter = 35;
tik.mu = 1e-12;
[rupre,outup] = UPRE(A,b,[d,1,1],tik,sigma);
% [rupsig,outupsig] = UPRE_sigma(A,b,[d,1,1],tik,1);% sigma*1000);
tik.mu = 1e-12;
[rupsig,outupsig] = UPRE(A,b,[d,1,1],tik,0);
% display
ee = norm(A*rec2-b)^2;
fprintf('m*sigma^2 = %g\n',m*sigma^2);
fprintf('||Au-b||_2^2 = %g\n',ee);
fprintf('||Au_{CG}-b||_2^2 = %g\n',norm(A*rec3-b)^2);
fprintf('rel difference = %g\n',myrel(m*sigma^2,ee));
%%
figure(53);
subplot(4,2,1);hold off;
plot(x,'linewidth',2);hold on;
plot(rec2,'linewidth',2);
title('DP soln');legend({'true','rFP iter recov'});
subplot(4,2,2);semilogy(out2.mu);
xlabel('iteration/10');ylabel('mu');
subplot(4,2,4);semilogy(out3.mu);
xlabel('iteration');ylabel('mu');
subplot(4,2,3);
plot(rec3,'linewidth',2);
title('DP soln with CG solve');
subplot(4,2,5);
plot(rupre,'linewidth',2);title('upre soln');
subplot(4,2,6);semilogy(outup.mus);
xlabel('iteration');ylabel('mu');title('upre');
subplot(4,2,7);
plot(rupsig,'linewidth',2);title('upre soln -sig est');
subplot(4,2,8);semilogy(outupsig.mus);
xlabel('iteration');ylabel('mu');title('upre w/ sig est');
figure(89);hold off;
semilogy(outupsig.sigmas);title('sigmas');xlabel('iteration');
hold on;plot(ones(numel(outupsig.sigmas),1)*sigma);
legend('guess sigmas','true');hold off;