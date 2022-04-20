% a simple demo for using Tikhonov regularization

d = 500; % signal dimension
m = 300; % number of samples
SNR = 50;    % SNR, with mean zero i.i.d. Gaussian noise
rng(2022);

% generate a simple test signal
x = sin(3*pi*linspace(0,1,d)');
A = randn(m,d); % random sampling matrix
% A = eye(d);

% generate samples and add noise
b = A*x;
b = b + randn(m,1)*mean(abs(b))/SNR;

% algorithm parameters
tik.mu = 5; % regularization balancing parameter
tik.order = 2;  % order of the finite differencing scheme
tik.iter = 5000; % maximum number of iterations
tik.tol = 1e-8; % convergence tolerance
tik.scale_A = true;
tik = check_tik_opts(tik);
V = my_Fourier_filters(tik.order,1,d,1,1);

R = @(u,sqlambda)ifftn(fftn(u)./(1+sqlambda^2*V));
opts.iter = tik.iter;
opts.sigma0 = (tik.mu)^(-1/2);
opts.tol = tik.tol;
% compute solution
[rec0,out0] = Tikhonov_SD(A,b,[d,1,1],tik);
[rec,out] = Tikhonov_CG(A,b,[d,1,1],tik);
[rec2,out2] = Tikhonov_Nesta(A,b,[d,1,1],tik);
[rec3,out3] = Tikhonov_prox(A,b,[d,1,1],tik);
[rec4,out4] = PnP3_prox(A,R,b,[d,1,1],opts);
% rtrue = ifft(fft(b)./(1+V/mu));
% myrel(rtrue,rec0)
myrel(rec0,rec)
myrel(rec0,rec2)
myrel(rec0,rec3)
myrel(rec0,rec4)

%%
% display
figure(53);tiledlayout(2,2);
nexttile;hold off;
plot(x,'linewidth',2);hold on;
plot(rec,'linewidth',2);
plot(rec2,':','linewidth',1.5);
plot(rec3,'--');
plot(real(rec4),'-.');
legend({'true','CG','GD NESTA','prox. grad.'});
hold off;
nexttile;hold off;
semilogy(out0.rel_chg);hold on;
semilogy(out2.rel_chg);
semilogy(out3.rel_chg);
semilogy(out4.rel_chg);
legend('BB','GD NESTA','prox. grad.')
hold off;