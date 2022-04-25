function [D,Dt] = penaltyFilter(regV)

D = @(U)ForwardFilt(U,regV);
Dt = @(U)AdjointFilt(U,regV);


function U = ForwardFilt(U,regV)

[p,q,r] = size(regV);
U = reshape(U,p,q,r);
U = col(ifftn(fftn(U).*regV));


function U = AdjointFilt(U,regV)

[p,q,r] = size(regV);
U = reshape(U,p,q,r);
U = col(ifftn(fftn(U).*conj(regV)));
