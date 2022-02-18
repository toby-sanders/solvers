function [x,out] = RL2DSubBPMF(h,g,b,M,opts)

if nargin<5
    opts.iter = 100;
end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if isfield(opts,'x'), x = opts.x; end
if ~isfield(opts,'disp'), opts.disp = 0; end
epsilon = 1e-8;


% bstr = double(M);
[p,q] = size(M);
k = size(b,2);
S = find(M);
bstr = zeros(p*q,k);
for i = 1:k, bstr(S,i) = b(:,i); end
bstr = reshape(bstr,p,q,k);

hhat = fft2(h,p,q);
ghat = fft2(g,p,q);
Ghat = ghat.*hhat;

if ~isfield(opts,'x')
% x = ifftn(Ghat.*fftn(bstr));
% x = bstr + epsilon;
x = ones(p,q)*100; %rand(p,q)*100;
end
Gx = ifft2(fft2(x).*Ghat);
normalizer = zeros(p,q);% ifft2(fft2(double(M)).*conj(Ghat(:,:,1)));
for i = 1:k
    normalizer = normalizer + ifft2(fft2(double(M)).*conj(Ghat(:,:,i)));
end
out.rel_error = [];
out.obj_f = [];
out.rel_M = [];
flg = 0; % only find the mask once
for i = 1:opts.iter
    xp = x;
    cfactor1 = bstr./(Gx+epsilon);
    cfactor = sum(ifft2(fft2(cfactor1).*conj(Ghat)),3)./(normalizer+epsilon);   
    x = x.*cfactor;
    out.rel_error = [out.rel_error;myrel(x,xp)];
    out.obj_f = [out.obj_f;sum(col(M.*(-Gx + bstr.*log(Gx))))];
    if mod(i,20) ==0 && flg ==0
        Mp = M;
        M = sum(bstr,3)./sum(Gx+epsilon,3);
        out.rel_M = [out.rel_M;myrel(M,Mp)];
        SS = M>2;
        M = ones(p,q);
        M(SS) = 0;
        bstr = reshape(bstr,p*q,k);
        normalizer = zeros(p,q);
        for j = 1:k
            bstr(SS,j) = 0;
            normalizer = normalizer + ifft2(fft2(double(M)).*conj(Ghat(:,:,j)));
        end
        x = ones(p,q)*100;
        bstr = reshape(bstr,p,q,k);
        flg = 1;
    end
    Gx = ifft2(fft2(x).*Ghat);
    if opts.disp
        M = sum(bstr,3)./sum(Gx+epsilon,3);
        figure(75);
         subplot(2,2,3);imagesc(real(x));
         title(['o, object Estimate, iteration = ',num2str(i)]);colorbar;
         subplot(2,2,4);mesh(min(real(M),7));
         title('data divided by data estimate');colorbar;
         if isfield(opts,'true')
             subplot(2,2,1);imagesc(opts.true);
             title('u, original image');colorbar;
         else
            subplot(2,2,1);imagesc(real(Gx(:,:,1)));
            title('G*x');colorbar;
         end
        subplot(2,2,2);imagesc(bstr(:,:,1))
         title('d, blurry image data with bad pixels');colorbar;
        % colormap(pink);
    end
%     pause;
end

out.M = M;