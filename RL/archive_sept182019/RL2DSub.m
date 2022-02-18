function [x,out] = RL2DSub(h,g,b,M,opts)

% in this version, only a subset of the data is available, which is located
% at the pixels in the mask M
% the main difference is that the normalization in the denomenator is not a
% constant

if nargin<5
    opts.iter = 100;
end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end
if isfield(opts,'x'), x = opts.x; end
if ~isfield(opts,'disp'), opts.disp = 0; end
epsilon = 1e-10;


bstr = double(M);
bstr(find(M)) = b;
[p,q] = size(M);
hhat = fftn(h,[p,q]);
ghat = fftn(g,[p,q]);
Ghat = ghat.*hhat;

if ~isfield(opts,'x')
% x = ifftn(Ghat.*fftn(bstr));
% x = bstr + epsilon;
x = ones(p,q)*100; %rand(p,q)*100;
end
Gx = ifftn(fftn(x).*Ghat);
nrmconst = ifftn(fftn(double(M)).*conj(Ghat));
out.rel_error = [];
out.obj_f = [];
for i = 1:opts.iter
    xp = x;
    cfactor1 = bstr./(Gx+epsilon);
    cfactor = ifftn(fftn(cfactor1).*conj(Ghat))./(nrmconst+epsilon);   
    x = x.*cfactor;
    Gx = ifftn(fftn(x).*Ghat);
    out.rel_error = [out.rel_error;myrel(x,xp)];
    out.obj_f = [out.obj_f;sum(col(M.*(-Gx + bstr.*log(Gx))))];
    if opts.disp
        figure(75);
        subplot(2,2,1);imagesc(x);title('x');colorbar;
        subplot(2,2,2);imagesc(cfactor);title('correction');colorbar;
        subplot(2,2,3);imagesc(Gx);title('G*x');colorbar;
        subplot(2,2,4);imagesc(bstr);title('b');colorbar;
    end
end