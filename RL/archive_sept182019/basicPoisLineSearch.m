function [tau,fval]= basicPoisLineSearch(Hu,Hg,b,gam)

% line search for Poisson log likelihood
% searching along step tau by
%      Unew = Up + tau(U-Up)
% return:
% tau - good estimate for step length
% fval - function value at that location

if nargin<4, gam = 2; end


sumHg = sum(Hg(:));
sumHu = sum(Hu(:));      
objFold = -sum(Hu(:)) + sum(b(:).*log(Hu(:)));
objFnew = -(sumHu+sumHg) + sum(b(:).*log(Hu(:) + Hg(:)));
tau = 1;
while objFnew>objFold & imag(sum(objFnew(:)))==0
   tau = tau*gam;
   objFold = objFnew;
   objFnew = -(sumHu+tau*sumHg) + sum(b(:).*log(Hu(:) + tau*Hg(:)));
end
tau = tau/gam;
fval = objFold;
