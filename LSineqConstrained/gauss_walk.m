d = 100;
etas = [.01 .1 1 100];
% dist_type = 'laplace';
dist_type = 'gauss';

x = zeros(d,numel(etas));

for j = 1:numel(etas)
    if strcmp(dist_type,'gauss');
        r = randn(d-1,1)*etas(j);
    elseif strcmp(dist_type,'laplace');
    % r = laprnd(
        r = log(rand(d-1,1)./rand(d-1,1))*etas(j);
    end
for i = 2:d
    x(i,j) = x(i-1,j)+r(i-1);
end
subplot(4,1,j);plot(x(:,j));
title(['eta = ',num2str(etas(j))]);
axis([1 d min(x(:,j)) max(x(:,j))]);
end