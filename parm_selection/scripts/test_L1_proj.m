% a simple demo for using Tikhonov regularization
clear;
d = 500; % signal dimension
m = 5000; % number of samples
SNR = 5;    % SNR, with mean zero i.i.d. Gaussian noise
order = 1; % order of the regularization

% generate a simple test signal
% x = sin(3*pi*linspace(0,1,d)');
x = zeros(d,1);x(d/4:3*d/4) = 1;
% A = randn(m,d); % random sampling matrix
A = eye(d); m = d;

% generate samples and add noise
b = A*x;
sigma = mean(abs(b))/SNR;
b = b + randn(m,1)*sigma;

% algorithm parameters
tik.mu = 1e-1; % regularization balancing parameter
tik.order = order;  % order of the finite differencing scheme
tik.iter = 100; % maximum number of iterations
tik.tol = 1e-5; % convergence tolerance
tik.maxiter = 35;

% compute solutions
[recup,outup] = UPRE_fixedpt(A,b,[d,1,1],tik,sigma);
lam = 1/outup.mus(end);
eta = sigma/sqrt(lam);
mu = eta/(sqrt(2)*sigma^2);

hotv.iter = 300;
hotv.tol = 1e-4;
hotv.order = order;
hotv.scale_A = false;
hotv.scale_mu = false;
hotv.mu = mu;
[recL1,outL1] = HOTV3D(A,b,d,hotv);

% [rupsig,outupsig] = UPRE_fixedpt_sigma(A,b,[d,1,1],tik,sigma*1000);
% display
ee = norm(A*recup-b)^2;
eel1 = norm(A*recL1-b)^2;
fprintf('\nm*sigma^2 = %g\n',m*sigma^2);
fprintf('||Au-b||_2^2 = %g\n',ee);
fprintf('||Au_{L1}-b||_2^2 = %g\n',eel1);
%%
figure(53);
subplot(1,2,1);hold off;
plot(x,'linewidth',2);hold on;
plot(recup,'linewidth',2);
plot(recL1,'--','linewidth',1.5);
title('DP soln');legend({'true','rFP iter recov','L1 proj'});
subplot(1,2,2);semilogy(outup.mus);
xlabel('iteration/10');ylabel('mu');