function [U,out] = inpaint_3D_iter(bb,S,Wf,d,opts)



% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 09/22/2016



if numel(d)<3
    d(end+1:3) = 1;
elseif numel(d)>3
    error('n can have at most 3 dimensions');
end
%p = d(1);q = d(2); r = d(3);

if size(S,2)==3
    S = sub2ind(d,S(:,1),S(:,2),S(:,3));
end

if numel(S)~=numel(bb)
    error('number of data points and specified indices dont match');
end


A = @(x,mode)subdata_select(x,mode,S,d,numel(Wf));

[U,out] = HOTV3D_ip_iter(A,[Wf;bb],numel(Wf),d,opts);








function x = subdata_select(x,mode,S,d,lWf)

switch mode
    case 1
        x = [x;x(S)];
    case 2
        y = zeros(d);
        y(S) = x(lWf+1:end);
        x = y(:)+x(1:lWf);
end