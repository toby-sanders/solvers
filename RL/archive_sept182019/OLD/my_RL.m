function [x,out,xr] = my_RL(h,b,n,opts)


if nargin<4
    opts.iter = 100;
end
if isfield(opts,'tol'), tol = opts.tol;
else, tol = 1e-7; end
if ~isfield(opts,'iter'), opts.iter = 100; end

epsilon = 1e-10;
lambda = 1e-5;

x = b;xr = x;
hhat = fft(h,n);
out.rel_error = [];
d = n(1);
T = zeros(n(1));T(1:d+1:end) = -1;T(d+1:d+1:end) = 1;
T(end,1) = 1;% T = T^2;
TtT = T'*T;
% TtT = eye(d);
for i = 1:opts.iter
    
   Hx = ifft(hhat.*fft(x));
   Hxr = ifft(hhat.*fft(xr));
   bDHx = (b+epsilon)./(Hx+epsilon);
   bDHxr = (b+epsilon)./(Hxr+epsilon);
   xp = x;
   x = x.*(ifft(conj(hhat).*fft(bDHx)));
   xr = xr.*(ifft(conj(hhat).*fft(bDHxr)))./(1+lambda*TtT*xr);% ./(1-lambda*[diff(xr,2);0;0]);
   xr = max(xr,0);
   out.rel_error = [out.rel_error;norm(x-xp)/norm(xp)];
   if out.rel_error(end)<tol, break;end
%    figure(75);
%    plot(x);title(i); 
%    pause;
end
out.iters = i;