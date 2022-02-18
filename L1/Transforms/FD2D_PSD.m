function [D,Dt] = FD2D_PSD(p,q,PSD)


% Written by Toby Sanders @Lickenbrock Tech.
% 8/6/2020

D = @(U)D_Forward(U,PSD,p,q);
Dt = @(U)D_Adjoint(U,PSD,p,q);


% high order finite differences
function U = D_Forward(U,PSD,p,q)
   
U = real(col(ifft2(fft2(reshape(U,p,q)).*PSD)));
    

%transpose FD
function U = D_Adjoint(U,PSD,p,q)

U = real(col(ifft2(fft2(reshape(U,p,q)).*conj(PSD))));


    
    
    
    