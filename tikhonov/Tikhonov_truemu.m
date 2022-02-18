function [X,out] = Tikhonov_truemu(A,b,n,opts,Utrue)

%opts.lambda = .02;
%stlam = .02;
opts.mu = 1e2;
stmu =  20;
out.err_true = 1;
out.mu = [];

for i = 1:1000
    [U,~] = Tikhonov_CG(A,b,n,opts);
    out.err_true = [out.err_true;myrel(U,Utrue,2)];
    out.mu = [out.mu; opts.mu];
    fprintf('true error = %g, mu = %g\n',out.err_true(end),opts.mu);
    if out.err_true(end)>out.err_true(end-1)
        if abs(stmu)<.0002
            out.mu_optimal = opts.mu - stmu;
            out.err_optimal = out.err_true(end-1);
            X = Up;
            break;
        end
        stmu = -stmu/2;        
    end
    opts.mu = opts.mu + stmu;
    Up = U;
    if opts.mu<=0
        opts.mu = opts.mu - stmu;
        stmu = stmu/4;
        opts.mu = opts.mu + stmu;
    end
end