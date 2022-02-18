d = 1024;
m = d/2;
SNR = 20;
N = 3;
k = 3;

x = zeros(d,1);
x(round(d/6):round(d/6+d/4)) = ...
    2-linspace(...
    -sqrt(2),sqrt(2),numel(round(d/6):round(d/6+d/4))).^2;
x(round(d/2):round(d/2+d/6)) = 1;
x(round(d/2+d/6):round(d/2+d/3)) ...
    = linspace(1,0,numel(round(d/2+d/6):round(d/2+d/3)));

% [D,Dt] = FD3D_platte(k,d,1,1,N);
% [flg] = check_D_Dt(D,Dt,[d,1,1]);
% 
% dx = D(x);dx = dx(:,2);
% hold off;
% plot(x);hold on;
% plot(dx);hold off;
% shg;



A = sprand(m,d,1/10);%randn(d*S_rate,d);
b = A*x;
b = b + randn(m,1)*mean(abs(b))/SNR;

opts.nonneg = true;
opts.mu = 20;
opts.max_iter = 300;
opts.order = k;
opts.levels = 1;

rec = HOTV3D_platte(A,b,[d,1,1],opts);
rec2 = HOTV3D(A,b,[d,1,1],opts);
hold off;
plot(rec);hold on;
plot(rec2);hold off;
shg;