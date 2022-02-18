function [X,out] = tikhonov_truelam(A,b,d1,d2,d3,opts,Utrue)

%opts.lambda = .02;
%stlam = .02;
opts.lambda = 4;
stlam = .5;
out.err_true = 1;


for i = 1:1000
    [U,~] = tikhonov_cgls(A,b,d1,d2,d3,opts);
    out.err_true = [out.err_true;myrel(U,Utrue,2)];
    fprintf('true error = %g, lambda = %g\n',out.err_true(end),opts.lambda);
    if out.err_true(end)>out.err_true(end-1)
        if abs(stlam)<.0002
            out.lam_optimal = opts.lambda - stlam;
            out.err_optimal = out.err_true(end-1);
            X = Up;
            break;
        end
        stlam = -stlam/2;        
    end
    opts.lambda = opts.lambda + stlam;
    Up = U;
end