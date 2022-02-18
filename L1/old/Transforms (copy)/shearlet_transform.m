function [Y] = shearlet_transform(X,shearSys,mode)


% Written by Toby Sanders @ASU
% Department of Statistical & Mathematical Sciences
% 05/17/2016



%compute shearlet coefficients 
switch mode
    case 1
        Y = zeros(size(shearSys.shearlets));
        Xfreq = fftshift(fft2(X));
        for i = 1:shearSys.nShearlets
            Y(:,:,i) = ifft2(ifftshift(Xfreq.*conj(shearSys.shearlets(:,:,i))))/shearSys.RMS(i);
        end
        Y = Y*mean(shearSys.RMS);
    case 2
        Y = zeros(size(X,1),size(X,2));
        %transform coefficients into frequency domain
        X = fft2(X);
        for i = 1:shearSys.nShearlets
            Y = Y + fftshift(X(:,:,i)).*shearSys.shearlets(:,:,i)/shearSys.RMS(i);
        end
        Y = ifft2(ifftshift(Y));
        Y = Y(:)*mean(shearSys.RMS);
        
end