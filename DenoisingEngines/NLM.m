function y = NLM(x,blockSize)


% non-local means image denoising
% written by Toby Sanders @ Lickenbrock technologies
% input noisy image - x
% output denoised image - y


if nargin<2
    blockSize = 4;
end
[p,q] = size(x);
y = zeros(size(x));
Z = zeros(blockSize,blockSize,p*q);
X = zeros(p+blockSize,q+blockSize);
X(1:p,1:q) = x;
X = circshift(X,[blockSize/2,blockSize/2]);

% construct matrix of all blocks
cnt = 0;
for i = 1:q
    for j = 1:p
        cnt = cnt+1;
        Z(:,:,cnt) = X(j:j+blockSize-1,i:i+blockSize-1);
    end 
end

% compute weights and average
h = .2;
for i = 1:p-blockSize
    for j = 1:q-blockSize
        cBlock = x(i:i+blockSize-1,j:j+blockSize-1);
        W = cBlock - Z;
        W = reshape(exp(-sum(sum(W.*W))/h),p,q);
        C = 1/sum(W(:));
        y(i+blockSize/2,j+blockSize/2) = C*sum(sum(W.*x));        
    end
end
