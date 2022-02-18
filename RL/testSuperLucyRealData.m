% multiframe blind deconvolution simulations
% a sensitivity is added to EVERY pixel given by MTRUE

clear;
d = 512;
iter = 150;
minV = 500; % min and max image values (important for Poisson noise level)
maxV = 1000;
bg = ones(d,d)*minV;
nF = 5;  % number of frames
Ntrials = 30; % number of updates on the sensitivity pixels
eta = 5; % sensitivity parameter (maximum of sensitivity)
lambda = 1e-4; % TV parameter
order = 2; % regularization order
levels = 3;
noise = true;   % add noise or no?
bpcluster = false;  % add a "cluster" of sensitive pixels?
perc = 0;   % percentage of "bad pixels"
superRes = false;
load('I_band.mat');

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
params.mask = ones(d,d);
params.bg = background;
params.lambda = 0;
[rec,out] = myLucy(h,b,params);
params.lambda = lambda;
params.order = 1;
params.levels = 1;
[recTV,outTV] = myLucy(h,b,params);
params.order = 2;
params.levels = 2;
[recH,outH] = myLucy(h,b,params);
mydir = pwd;
cd('C:\Users\toby.sanders\Documents\repos\PySkyMat');

recGC = glintCorrection_wf_script(rec,1e-3,.5,5);
recTVGC = glintCorrection_wf_script(recTV,1e-3,.5,5);
recHGC = glintCorrection_wf_script(recH,1e-3,.5,5);
dataGC = glintCorrection_wf_script(max(b-bg,0),1e-3,.5,5);
cd(mydir);
%%
arange = [170,350,170,350];
gam = .5;
figure(99);
subplot(2,2,1);imagesc(recGC.^gam);title('rec lambda =0');colorbar;
axis(arange);
subplot(2,2,3);imagesc(recTVGC.^gam);title('rec TV');colorbar;
axis(arange);
subplot(2,2,2);semilogy(out.rel_chg);title('rel change in sol');
subplot(2,2,4);imagesc(recHGC(:,:,1).^gam);title('MHOTV');
axis(arange);colorbar;
colormap(gray);


