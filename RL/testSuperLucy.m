% multiframe blind deconvolution simulations
% a sensitivity is added to EVERY pixel given by MTRUE

clear;
d = 300;
iter = 200;
minV = 500; % min and max image values (important for Poisson noise level)
maxV = 1000;
bg = ones(d,d)*minV;
nF = 5;  % number of frames
Ntrials = 30; % number of updates on the sensitivity pixels
eta = 5; % sensitivity parameter (maximum of sensitivity)
lambda = 1e-3; % TV parameter
order = 2; % regularization order
levels = 3;
noise = true;   % add noise or no?
bpcluster = false;  % add a "cluster" of sensitive pixels?
perc = 0;   % percentage of "bad pixels"
superRes = false;

% random Gauss convolution kernels, set min and max sigma
maxsig = 0.01;% 1e-3;  
minsig = 0.01;% 1e-3;
sigmas = rand(nF,1)*(maxsig- minsig) + minsig;

% x = phantom(d);
% M = ones(d);
% x = imresize(double(imread('blobs.png')),[d,d]);
% x = imresize(double(rgb2gray(imread('peppers.png'))),[d,d]);
x = imresize(double(imread('satelliet.png')),[d,d]);
x = x/max(col(x))*(maxV-minV);

% generate random sensitivity matrix
S = rand(d^2,1);
[~,S] = sort(S);
S = sort(S(1:round(d^2*perc)));
S2 = rand(numel(S),1);
[~,S2] = sort(S2);
S2 = S2(1:round(numel(S2)/2));
Mtrue = ones(d,d);
Mtrue(S) = rand(numel(S),1)*(eta-1) + 1;
Mtrue(S(S2)) = 1./Mtrue(S(S2));
if bpcluster   % add a cluster of sensitive pixels
    Mtrue(144:147,144:147) = .4;
end
M = ones(d,d); % bad pixel mask

% construct the convolutional kernels
h = zeros(d,d,nF);
hb = zeros(d,d,nF);
b = zeros(d^2,nF);
[X,Y] = meshgrid(linspace(-1,1,d),linspace(-1,1,d));
for i = 1:nF
    if sigmas(i) ~=0       
        h(:,:,i) = exp(-(X.^2+Y.^2)/sigmas(i)^2)';
        h(:,:,i) = fftshift(h(:,:,i)/sum(sum(h(:,:,i))));
    else
        h = 1;
    end
    hb(:,:,i) = my_conv3D(x,h(:,:,i));
    tmp = hb(:,:,i);
    b(:,i) = tmp(:);
    tmp(:) = 0;
    tmp(M) = b(:,i);
end
b = reshape(b,d,d,nF).*Mtrue + bg;
if noise, b = imnoise(b*1e-12,'poisson')*1e12; end

% parameters for RL algorithm
params.nonneg = true;
params.iter = iter;
params.tol = 1e-7;
params.lambda = lambda;
params.order = order;
params.superRes = superRes;
params.levels = levels;

% intialize variables for loop
recMF = zeros(d,d,Ntrials);
rel_chg = zeros(Ntrials-1,1);
eeM = zeros(Ntrials,1);
eeX = zeros(Ntrials,1);
ratio = zeros(d,d,Ntrials);
ratioCumm = zeros(d,d,Ntrials);
dataEst = zeros(d,d,nF);
mEst = ones(d,d,Ntrials+1);
params.mask = Mtrue;
params.bg = bg;
[rec,out] = myLucy(h,b,params);

%%
figure(99);
subplot(2,2,1);imagesc(rec);title('rec');colorbar;
subplot(2,2,2);semilogy(out.rel_chg);title('rel change in sol');
subplot(2,2,3);plot(out.objF);title('objective function value');
subplot(2,2,4);imagesc(b(:,:,1)-bg);title('blurry noisy data');colorbar;
colormap(gray);


