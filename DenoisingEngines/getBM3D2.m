function R = getBM3D(sigma,scl)

if nargin<1, sigma = 5/255; end
R = @(z)myBM3Dcaller(z,sigma,scl);

function z = myBM3Dcaller(z,sigma,scl)

% scl = max(z(:));
z = BM3D(z/scl,sigma);
z = z*scl;
    