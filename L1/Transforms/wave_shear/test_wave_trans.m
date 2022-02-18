% toying back around with the wavelet operators
% turns out the wavelet rec filters are the same (flipped) of the wavelet
% dec filters.  Actually it turns out that you can just use conjugate
% filtering instead (f*conj(g)).  That is, if T is a conv. oper (circulant
% matrix), s.t. Tx = h*x, then T^Tx = conj(h)*x = \tilde h * x, where
% \tilde h is a flipped version of h.

clear;
n = 50;
SNR = 4;
order = 1;
levels = 3;
th = 0.1;

wname = ['db',num2str(order)];
x = zeros(n,1);
x(30:40) = 1;
% x = sin(2*pi*linspace(0,1,n)');
x = x + randn(n,1)*mean(abs(x))/SNR;

[Lodec,Hidec,Lorec,Hirec] = wfilters(wname);
[D,Dt] = my_wav_trans_1D(wname,levels);

tic;
% wavelet decomposition
xc = zeros(n,levels+1);
xd = x;
for i = 1:levels
    xc(:,i) = my_conv3D(xd,Hidec');
    xd = my_conv3D(xd,Lodec');
end
xc(:,end) = xd;

% denoising/thresholding
xco = xc;
xc(:,1:end-1) = sign(xc(:,1:end-1)).*max(0,abs(xc(:,1:end-1))-th);
xc(find(abs(xc(:,end))<th),end)=0; 
xcoo = xc;

% reconstruction
xcrc = zeros(n,levels);
for i = 1:levels
    xcrc(:,i) = my_cconv3D(xc(:,i),Hidec')/2;
    for j = i+1:levels+1
        xc(:,j) = my_cconv3D(xc(:,j),Lodec')/2;
    end
end
xcr = sum(xcrc,2);
xdr = xc(:,end);
xrec = sum(xcrc,2)+xc(:,end);%  circshift(xcr+xdr,0);.
toc;



figure(74);
hold off;
plot(xrec);hold on;
plot(x);
legend('denoised','noisy');
% plot(xdr);
hold off;
% subplot(4,1,2);
% plot(Dt(D(x))+xdr);
% subplot(4,1,3);
% plot(xcrc);

figure(85);
plot(xco);
myrel(xrec,x)
figure(86);
plot(xcoo);
% figure(75);
% subplot(3,1,1);plot(c(:));hold on;
% % subplot(3,1,1);plot(c2(:));
% hold off;
% subplot(3,1,3);hold off;
% plot(linspace(0,1,100),imresize(Lodec(:),[100,1],'nearest'),'linewidth',2);
% hold on;
% plot(linspace(0,1,100),imresize(Hidec(:),[100,1],'nearest'),':','linewidth',2);
% hold off;
% 
% 
% figure(76);hold off;
% plot(x);hold on;
% plot(my_conv3D(x,Lodec'));hold off;