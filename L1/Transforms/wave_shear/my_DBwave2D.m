function U = my_DBwave2D(U,order,levels,th)

% get wavelet filters and tranform problem to Fourier space
[p,q] = size(U);
[lod,hid] = wfilters(['db',num2str(order)]);
lox = fft(lod',q);
loy = fft(lod',p);
hix = fft(hid',q);
hiy = fft(hid',p);
[LOX,LOY] = meshgrid(lox,loy);
[HIX,HIY] = meshgrid(hix,hiy);
Ux = fft(U,q,2);
Uy = fft(U,p,1);

% decomposition (in X and Y separatley)
Uc = zeros(p,q,levels+1,2);
Uc(:,:,1,1) = Ux.*HIX;
Uc(:,:,1,2) = Uy.*HIY;
for i = 2:levels
    Uc(:,:,i,1) = Uc(:,:,i-1,1).*LOX;
    Uc(:,:,i,2) = Uc(:,:,i-1,2).*LOY;
end
Uc(:,:,end,1) = Ux.*(LOX.^levels);
Uc(:,:,end,2) = Uy.*(LOY.^levels);

% denoising/ thresholding
Uc(:,:,:,1) = ifft(Uc(:,:,:,1),q,2); % go to real space
Uc(:,:,:,2) = ifft(Uc(:,:,:,2),p,1);
Uc(:,:,1:end,:) = max(0,abs(Uc(:,:,1:end,:))-th).*sign(Uc(:,:,1:end,:));
Uc(:,:,:,1) = fft(Uc(:,:,:,1),q,2); % back to Fourier
Uc(:,:,:,2) = fft(Uc(:,:,:,2),p,1); 


% reconstruction
for i = 1:levels
    Uc(:,:,i,1) = Uc(:,:,i,1).*conj(HIX.*(LOX.^(i-1)))/2^i;
    Uc(:,:,i,2) = Uc(:,:,i,2).*conj(HIY.*(LOY.^(i-1)))/2^i;
end
Uc(:,:,end,1) = Uc(:,:,end,1).*conj(LOX.^levels)/2^levels;
Uc(:,:,end,2) = Uc(:,:,end,2).*conj(LOY.^levels)/2^levels;
% sum and go back to real space
U = ifft(sum(Uc(:,:,:,1),3),q,2)/2+ifft(sum(Uc(:,:,:,2),3),p,1)/2;




