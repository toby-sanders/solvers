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
th = 1e-1;

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
FLodec = fftn(Lodec',[n,1,1]);
FHidec = fftn(Hidec',[n,1,1]);
Fx = fftn(x,[n,1,1]);
xc(:,1) = Fx.*FHidec;
for i = 2:levels
    % xc(:,i) = my_conv3D(xd,Hidec');
    % xd = my_conv3D(xd,Lodec');
    % xc(:,i) = Fx.*FHidec.*(FLodec.^(i-1));
    xc(:,i) = xc(:,i-1).*FLodec;
end
xc(:,end) = Fx.*(FLodec.^(levels));


% denoising/thresholding
xc = ifft(xc);  % go to real space
xc(:,1:end-1) = sign(xc(:,1:end-1)).*max(0,abs(xc(:,1:end-1))-th);
xc = fft(xc);   % back to Fourier!!

% reconstruction
xcrc = zeros(n,levels);
for i = 1:levels
%     xcrc(:,i) = my_cconv3D(xc(:,i),Hidec')/2;
%     for j = i+1:levels+1
%         xc(:,j) = my_cconv3D(xc(:,j),Lodec')/2;
%     end
    xcrc(:,i) = xc(:,i).*conj(FHidec.*(FLodec.^(i-1)))/2^i;
end
% xcr = real(sum(ifft(xcrc),2));
% xdr = ifft(xc(:,end).*conj(FLodec.^(levels)))/2^levels;
xrec = ifft(xc(:,end).*conj(FLodec.^(levels))/2^levels + sum(xcrc,2));%  circshift(xcr+xdr,0);
toc;

tic;
xrec2 = my_DBwave1D(x,order,levels,th);
toc;

tic;
xrec3 = my_DBwave2D(x,order,levels,th);
toc;

figure(74);
subplot(4,1,1);hold off;
plot(xrec);hold on;
plot(x);
 % plot(xdr);
 hold off;
subplot(4,1,2);
plot(Dt(D(x)));
subplot(4,1,3);
plot(ifft(xcrc));

figure(72);hold off;
plot(x);hold on;
plot(xrec2);

myrel(xrec,xrec2)
myrel(xrec2,xrec3)
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