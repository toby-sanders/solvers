function [U,out] = inpaint_3D(bb,S,d,opts)


% An algorithm for L1 optimized inpainting
% bb holds known image values
% S are the indices of bb (either vectorized or as triplets)
% d is the dimension (3D vector)
% opts are options for HOTV3D algorithm
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 09/22/2016



if numel(d)<3
    d(end+1:3) = 1;
elseif numel(d)>3
    error('d can have at most 3 dimensions');
end
%p = d(1);q = d(2); r = d(3);

if size(S,2)==3
    S = sub2ind(d,S(:,1),S(:,2),S(:,3));
end

if numel(S)~=numel(bb)
    error('number of data points and specified indices dont match');
end


A = @(x,mode)subdata_select(x,mode,S,d);

% determine an initial solution using much faster L2 formulation
if ~isfield(opts,'init')
    tik.lambda = .15;
    tik.maxit = 200;
    tik.tol = 1e-5;
    tik.order = opts.order;
    [opts.init,~] = Tikhonov(A,bb,d,tik);
end

% final solution
[U,out] = HOTV3D(A,bb,d,opts);
out.init = opts.init;








function x = subdata_select(x,mode,S,d)

switch mode
    case 1
        x = x(S);
    case 2
        y = zeros(d);
        y(S) = x;
        x = y(:);
end