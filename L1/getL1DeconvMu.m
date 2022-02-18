function [mu,out] = getL1DeconvMu(h,b,opts)

% simple code for automating the mu value for the L1 regularized
% deconvolution code, using the L2 regularized deconvolution with ME
% parameter selection

% The approach is described in the article
% Sanders, Toby, Rodrigo B. Platte, and Robert D. Skeel. 
% "Effective new methods for automated parameter selection in regularized 
% inverse problems." Applied Numerical Mathematics 152 (2020): 29-48.

% written by Toby Sanders @Lickenbrock Tech.
% 9/18/2020

% set parameters for L2 optimization
parm.order = opts.order;
parm.levels = opts.levels;
parm.tol = 1e-4;
parm.iter = 50;
parm.theta = 1;
[u,out] = HOTVL2_deblur(h,b,parm); % run L2 optimization and parm selection

% project the L2 parameter onto the L1 parameter by way of the MAP
% interpretation of the problems
mu = sqrt(out.etas(end))/(out.sigmas(end)*sqrt(2));