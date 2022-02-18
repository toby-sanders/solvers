function [tau,fval]= TVPoisLineSearch(Hu,Hg,up,g,b,lam,gam)

% line search for Poisson log likelihood
% searching along step tau by
%      Unew = Up + tau(U-Up)
% return:
% tau - good estimate for step length
% fval - function value at that location

% objective function value is
% sum(-Hu - Hg + b.*log(Hu + Hg)) - lam*TV(u)
if nargin<5, gam = 2; end

sumHg = sum(Hg(:));
sumHu = sum(Hu(:));      
objFold = -sum(Hu(:)) + sum(b(:).*log(Hu(:))) - computeTVnorm(up,lam);
objFnew = -(sumHu+sumHg) + sum(b(:).*log(Hu(:) + Hg(:))) - computeTVnorm(up+g,lam);
tau = 1;
if objFnew>objFold % forward line search
    while objFnew>objFold & imag(sum(objFnew(:)))==0
       tau = tau*gam;
       objFold = objFnew;
       objFnew = -(sumHu+tau*sumHg) + sum(b(:).*log(Hu(:) + tau*Hg(:)))...
           - computeTVnorm(up+tau*g,lam);
    end
    tau = tau/gam;
else % backtracking
    while objFnew<objFold & imag(sum(objFnew(:)))==0
       tau = tau/gam;
       objFnew = -(sumHu+tau*sumHg) + sum(b(:).*log(Hu(:) + tau*Hg(:)))...
           - computeTVnorm(up+tau*g,lam);        
    end
end

fval = objFold;


function uTV = computeTVnorm(u,lam)

ux = diff(u,1,2);
uy = diff(u,1,1);
uTV = lam*sum(sum(sqrt(ux(1:end-1,:).^2 + uy(:,1:end-1).^2)));