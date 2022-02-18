% a simple demo for using Tikhonov regularization
clear;
d = 500; % signal dimension
m = d; % number of samples
SNR = 1;    % SNR, with mean zero i.i.d. Gaussian noise
order = 2; % order of the regularization

% generate a simple test signal
x = sin(3*pi*linspace(0,1,d)');
% x = zeros(d,1);x(d/4:3*d/4) = 1;
% A = randn(m,d); % random sampling matrix
A = eye(d);

% generate samples and add noise
b = A*x;
sigma = mean(abs(b))/SNR;
b = b + randn(m,1)*sigma;

% algorithm parameters
tik.order = order;  % order of the finite differencing scheme
tik.tol = 1e-16; % convergence tolerance
tik.iter = 100;
tik.maxiter = 35;
tik.mu = 1e-12;
[rupre,outup] = UPRE(A,b,[d,1,1],tik,sigma);
[rupsig,outupsig] = UPRE_denoise(b,[d,1,1],tik,0);
%%
figure(53);
% subplot(4,2,1);hold off;
% plot(x,'linewidth',2);hold on;
% plot(rec2,'linewidth',2);
% title('DP soln');legend({'true','rFP iter recov'});
% subplot(4,2,2);semilogy(out2.mu);
% xlabel('iteration/10');ylabel('mu');
% subplot(4,2,4);semilogy(out3.mu);
% xlabel('iteration');ylabel('mu');
subplot(4,2,3);
plot(b,'linewidth',2);
title('noisy');
subplot(4,2,5);
plot(rupre,'linewidth',2);title('upre soln');
subplot(4,2,6);semilogy(outup.mus);
xlabel('iteration');ylabel('mu');title('upre');
subplot(4,2,7);
plot(rupsig,'linewidth',2);title('upre exact denoise');
subplot(4,2,8);semilogy(outupsig.mus);
xlabel('iteration');ylabel('mu');title('upre exact denoise');
figure(89);hold off;
semilogy(outupsig.sigmas);title('sigmas');xlabel('iteration');
hold on;plot(ones(numel(outupsig.sigmas),1)*sigma);
legend('guess sigmas','true');hold off;