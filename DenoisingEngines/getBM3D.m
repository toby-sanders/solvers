function R = getBM3D

R = @(z,sigma)myBM3Dcaller(z,sigma);

function z = myBM3Dcaller(z,sigma)

z = BM3D(z,sigma);
    