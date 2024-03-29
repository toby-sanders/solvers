function [flg,rel_diff] = check_A(A,N)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 08/24/2016

% Check if Dt is the adjoint of D
% If it is the true adjoint, then we should have x=y
% flg returns true if Dt is true adjoint and false otherwise
u = rand(N);% + 1i*rand(N); 

% smooth the random signal a tiny bit
if size(u,2)==1 && size(u,1)>100
    h = ones(50,1)/50;
    u = imfilter(u,h);
end

Du = A(u,1);

v = rand(size(Du));%  + 1i*rand(size(Du));

Dtv = A(v,2);

x = sum(Du(:).*conj(v(:)));
y = sum(u(:).*conj(Dtv(:)));

tol = 5e-2;

rel_diff = abs(x-y)/abs(x);
flg = rel_diff<tol;

