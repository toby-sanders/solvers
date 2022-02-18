function R = getGBM3D(opts)

if nargin<1, opts.profile = 'default'; end

R = @(z,sigma)GBM3D_distributed(z,sigma,opts);