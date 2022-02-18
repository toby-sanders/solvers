% toying back around with the wavelet operators in 2D

clear;
n = 300;
SNR = 2;
order = 1;
levels = 4;
th = .2;


P = phantom(n);
Pn = P + randn(n)*mean(abs(P(:)))/SNR;
tic;
U = my_DBwave2D(Pn,order,levels,th);
toc;
opts.order = order;
opts.max_iter = 250;
opts.mu = 10;
tic;
[U2,out] = HOTV3D(speye(n^2),Pn(:),[n,n,1],opts);
toc;
tic;
[U3,out3] = DBL1(speye(n^2),Pn(:),[n,n,1],opts);
toc;
myrel(U,Pn)
myrel(Pn,P)
myrel(U,P)
myrel(U2,P)

figure(77);
subplot(2,2,1);
imagesc(P,[0 1]);
subplot(2,2,2);imagesc(Pn,[0 1]);
subplot(2,2,3);imagesc(U,[0 1]);
subplot(2,2,4);imagesc(U2,[0 1]);