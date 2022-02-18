function [omega,out] = SURE_deblur(I,opts)

% an automated approach to parametric deconvolution using the SURE
% criterion.  For different regularization parameters (lambda), and
% different variance widths of a Gaussian kernel (omega^2), the optimal
% Gaussian kernel is determined by evaluating SURE (Stein's unbiased risk
% estimator) for simple Weiner filtered solutions.  The solutions are not
% "optimal," but this approach is used because it is very fast, and we only
% want to determine the kernel width

% written by Toby Sanders @Lickenbrock Tech.
% 11/22/2019


if ~isfield(opts,'lambdas'), lambdas = [1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4];
else, lambdas = opts.lambdas;
end
if ~isfield(opts,'omegas'), omegas = linspace(.1,4,20);
else, omegas = opts.omegas;
end
if ~isfield(opts,'V'), V = 1;
else, V = opts.V;
end

% very important to know the noise level for SURE!!!
% I have found the approach in this code works reasonably well
sigma = determineNoise(I); 

% initialize...
nO = numel(omegas); nL = numel(lambdas);
out.SURE = zeros(nO,nO,nL);
sBest = 1e20;
Ihat = fft2(I);
[p,q] = size(Ihat);

for i1 = 1:nO % loop over omegas (kernel width)
for i2 = 1:nO
    % construct new kernel and Fourier transform
    g = makeGausPSF2D([p,q],omegas(i1),omegas(i2));
    ghat = fft2(g);
    ghat2 = ghat.*conj(ghat);
    for j = 1:nL  % loop over lambdas (regularization parameter)
        lambda = lambdas(j);      
        % only need fourier transform of reconstruction.  The whole problem
        % stays in Fourier domain... only initial fft's are needed
        rec = Ihat.*conj(ghat)./(ghat2 + V*lambda); 
        Aub = col(rec.*ghat-Ihat);
        out.SURE(i1,i2,j) = Aub'*Aub/p/q + 2*sigma^2*sum(col(ghat2./(ghat2 + lambda*V)));
        if out.SURE(i1,i2,j)<sBest  % check for the minimal SURE
            sBest = out.SURE(i1,i2,j);
            out.rBest = rec;
            omega = [omegas(i1),omegas(i2)];
        end
    end
end
end
out.rBest = real(ifft2(out.rBest)); % move solution to real domain
out.SURE = out.SURE - p*q*sigma^2; % add constant to estimator
out.sigma = sigma;