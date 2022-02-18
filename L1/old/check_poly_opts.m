function opts = check_poly_opts(opts)


%original code by Chengbo Li

%Written by: Toby Sanders
%Comp. & Applied Math Dept., Univ. of South Carolina
%Dept. of Math and Stat Sciences, Arizona State University
%02/26/2016


%order of the finite difference operator
%
if ~isfield(opts,'order')
    fprintf('Order of finite difference set to 1\n');
    opts.order=1;
elseif round(opts.order)~=opts.order
    error('opts.order should be an integer');
elseif opts.order<0
    error('opts.order should be at least 0');
else
    fprintf('Using order %i regularization\n',opts.order);
end


if ~isfield(opts,'adaptive')
    opts.adaptive=false;
end


if isfield(opts,'scale_mu')
    if ~islogical(opts.scale_mu);
        error('opts.scale_mu should be true or false.');
    end
else
    opts.scale_mu = true;
end
% scale mu according to the order of PA transform


%mu is the most important parameter
% mu is mainly decided by noise level. Set mu big when b is noise-free
% whereas set mu small when b is very noisy.
if isfield(opts,'mu')
    if ~isscalar(opts.mu) || opts.mu <0
        error('opts.mu must be positive.');
    elseif opts.scale_mu
        if opts.mu < 24  || opts.mu > 201
            warning('opts.mu is not within optimal range');
        end
    else
        if opts.mu*2^(1-opts.order) < 24 || opts.mu*2^(1-opts.order)> 201
            warning('opts.mu is not within optimal range');
        end
    end
else
    %default mu
    if opts.scale_mu
        opts.mu = 90;
    else
        opts.mu = 90*2^(opts.order-1);
    end
end



% initial mu for continuation scheme
if isfield(opts,'mu0')
    if ~isscalar(opts.mu0) || opts.mu0 <= 0
        error('opts.mu0 is should be a positive number which is no bigger than beta.');
    end
else
    opts.mu0 = opts.mu/4;  
end




if isfield(opts,'scale_beta')
    if ~islogical(opts.scale_beta);
        error('opts.scale_beta should be true or false.');
    end
else
    opts.scale_beta = true;
end


% scale mu according to the order of PA transform
%coefficient for sparsifying operator
%setting beta = 2^5 usually works
%just need to work for mu
if isfield(opts,'beta')
    if ~isscalar(opts.beta) || opts.beta <0
        error('opts.beta must be positive.');
    else if opts.beta > 2^13 || opts.beta < 2^4
            warning('opts.beta is not within optimal range');
        end
    end
else
    opts.beta = 2^5;
    %opts.beta = opts.mu/16;
end



% initial beta
if isfield(opts,'beta0')
    if ~isscalar(opts.beta0) || opts.beta0 <= 0
        error('opts.beta0 is should be a positive number which is no bigger than beta.');
    end
else
    opts.beta0 = opts.beta; 
end



%option for wrap around regularization
if ~isfield(opts,'wrap_shrink')
    opts.wrap_shrink = false;
elseif opts.wrap_shrink
    warning('Using wrap around regularization');
end
    
% maxin gives the number of iterations (alternating gradient decent with
% shrinkage) before the multipliers are updated
if isfield(opts,'maxin')
    if ~isscalar(opts.maxin) || opts.maxin <= 0
        error('opts.maxin should be a positive integer.');
    end
else
    opts.maxin = 15;
end



% continuation parameter for mu and beta
% after each outer iteration, mu = min(muf , mu*opts.rate_ctn)
if isfield(opts,'rate_ctn')
    if ~isscalar(opts.rate_ctn) || opts.rate_ctn <= 1
        error('opts.rate_ctn is either not a scalar or no bigger than one.');
    end
else
    opts.rate_ctn = 1.5;
end





%If the signal is known to be nonnegative, this constraint can be set to
%true for usually better results
if isfield(opts,'nonneg')
    if ~islogical(opts.nonneg)
        error('opts.nonneg should be true or false.');
    end
else
    opts.nonneg = false;
end

%set max_c to true to implement a maximum signal size constraint
if ~isfield(opts,'max_c')
    opts.max_c = false;
elseif opts.max_c
    if ~isfield(opts,'max_v');
        error('maximum constraint was set to true without specifying the maximum value');
    end
end
    



% outer loop tolerence
%tolerance for convergence
%default is usually okay
if isfield(opts,'tol')
    if ~isscalar(opts.tol) || opts.tol <= 0
        error('opts.tol should be a positive small number.');
    end
else
    opts.tol = 1.e-3;
end;





% inner loop tolerence
%tolerance for updating parameters
%default is usually fine
if isfield(opts,'tol_inn')
    if ~isscalar(opts.tol_inn) || opts.tol_inn <= 0
        error('opts.tol_inn should be a positive small number.');
    end
else
    opts.tol_inn = 1.e-3;
end;




%For better convergence to a true minimum, increase the number of maxcnt.
%10 is usually okay and more than 30 is not needed
if isfield(opts,'maxcnt')
    if ~isscalar(opts.maxcnt) || opts.maxcnt <= 0
        error('opts.maxcnt should be a positive integer.');
    end
else
    opts.maxcnt = 10;
end





if isfield(opts,'maxit')
    if ~isscalar(opts.maxit) || opts.maxit <= 0
        error('opts.maxit should be a positive integer.');
    end
else
    opts.maxit = 1025;
end





%if the user has an initial guess, store it in this option
if isfield(opts,'init')
    if length(opts.init) ~= 1
        fprintf('User has supplied opts.init as initial guess solution!!!\n');
    elseif ~isinInterval(opts.init,0,1,true) || opts.init ~= floor(opts.init)
        error('opts.init should be either 0/1 or an initial guess matrix.');
    end
else
    opts.init = 1;
end





%option for displaying iteration information and final results
if isfield(opts,'disp')
    %if ~islogical(opts.disp)
    %    error('opts.disp should be true or false.');
    %end
else
    opts.disp = false;
end







%scaling of the operator A and vector b, scaling HIGHLY recommended so that
%consistent values for mu and beta may be used independent of the problem
if isfield(opts,'scale_A')
    if ~islogical(opts.scale_A)
        error('opts.scale_A should be true or false.');
    end
else
    opts.scale_A = true;
end


if isfield(opts,'scale_b')
    if ~islogical(opts.scale_b)
        error('opts.scale_b should be true or false.');
    end
else
    opts.scale_b = true;
end





%if the data is complex, such as fourier data, but the signal is real, set
%this option to true to search for a real solution
if isfield(opts,'isreal')
    if ~islogical(opts.isreal)
        error('opts.isreal should be true or false.');
    end
else
    opts.isreal = false;
end


%if isfield(opts,'TVL2')
%    if ~islogical(opts.TVL2)
%        error('opts.TVL2 should be true or false.');
%    end
%else
%    opts.TVL2 = false;
%end
% Decide the model: TV or TV/L2. The default is TV model, which is recommended.



%The remaining parameters are for the gradient decent
if isfield(opts,'c')
    if ~isscalar(opts.c) || opts.c <= 0 || opts.c > 1
        error('opts.c should be a scalar between 0 and 1.');
    end
else
    opts.c = 1.e-5;
end


if isfield(opts,'gamma')
    if ~isscalar(opts.gamma) || opts.gamma <= 0 || opts.gamma > 1
        error('opts.gamma should be a scalar between 0 and 1.');
    end
else
    opts.gamma = .6;
end


if isfield(opts,'gam')
    if ~isscalar(opts.gam) || opts.gam <= 0 || opts.gam > 1
        error('opts.gam should be a scalar between 0 and 1.');
    end
else
    opts.gam = .9995;
end
% Control the degree of nonmonotonicity. 0 corresponds to monotone line search.
% The best convergence is obtained by using values closer to 1 when the iterates
% are far from the optimum, and using values closer to 0 when near an optimum.




if isfield(opts,'rate_gam')
    if ~isscalar(opts.rate_gam) || opts.rate_gam <= 0 || opts.rate_gam > 1
        error('opts.rate_gam should be a scalar between 0 and 1.');
    end
else
    opts.rate_gam = .9;
end
% shrinkage rate of gam


%tau is the step length in the gradient decent
if isfield(opts,'tau')
    if ~isscalar(opts.tau) || opts.tau <= 0
        error('opts.tau is not positive scalar.');
    end
else
    opts.tau = 1.8;
end
opts.StpCr=0;

