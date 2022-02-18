function U = my_DBwave1D(U,order,levels,th)

% get filters and transform everything into Fourier space
[p,~] = size(U);
[lod,hid] = wfilters(['db',num2str(order)]);
lod = fft(lod',p);
hid = fft(hid',p);
U = fft(U);

% wavelet decomposition
Uc = zeros(p,levels+1);
Uc(:,1) = U.*hid;
for i = 2:levels
    Uc(:,i) = Uc(:,i-1).*lod;
end
Uc(:,end) = U.*(lod.^levels);

% denoising/thresholding
Uc = ifft(Uc); % go to real space
Uc(:,1:end) = max(0,abs(Uc(:,1:end))-th).*sign(Uc(:,1:end));
for i = 1:levels
    vec = 1:p;
    tmp = 1:2^i:p;
    vec(tmp) = '';
    Uc(vec,i) = 0;
end
% vec = 1:p;
% tmp = 1:2^(i-1):p;
% vec(tmp) = '';
Uc(vec,end) = 0;
nnz(Uc)
Uc = fft(Uc); % back to Fourier

% reconstruction from denoised coefficients
for i = 1:levels
    Uc(:,i) = Uc(:,i).*conj(hid.*(lod.^(i-1)));
end
Uc(:,end) = Uc(:,end).*conj(lod.^levels);
U = ifft(sum(Uc,2)); % add and back to real space


