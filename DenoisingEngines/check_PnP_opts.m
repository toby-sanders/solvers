function opts = check_PnP_opts(opts)

% This function checks the options for the PnP3 code.
% Improperly labeled field options will be identified, and all
% unlabeled fields are set to default values
%
% Written by Toby Sanders @Lickenbrock Tech.
% 10/22/2019

if ~isfield(opts,'sigma0')
    opts.sigma0 = 10/255;
end

% can simply specify iter to avoid selecting inner and outer iterations
if isfield(opts,'iter')
    if ~isscalar(opts.iter) || opts.iter <= 0
        error('opts.iter should be a positive integer.');
    end
else, opts.iter = 50; end

% number of inner loop iterations
% inner_iter gives the number of iterations for the minimization over u
if isfield(opts,'inner_iter')
    if ~isscalar(opts.inner_iter) || opts.inner_iter <= 0
        error('opts.inner_iter should be a positive integer.');
    end
else, opts.inner_iter = 3; end

% mu is generally the most important parameter
% mu is mainly decided by noise level. Set mu big when b is noise-free
% whereas set mu small when b is very noisy.
if isfield(opts,'mu')
    if ~isscalar(opts.mu) || opts.mu <0
        error('opts.mu must be positive.');
    end
else, opts.mu = 1e2; end

% important coefficient for the splitting variable
if isfield(opts,'beta')
    if ~isscalar(opts.beta) || opts.beta <0
        error('opts.beta must be positive.');
    end
else, opts.beta = 25; end

% convergence tolerance
if isfield(opts,'tol')
    if ~isscalar(opts.tol) || opts.tol<=0 || opts.tol>.1
        error('opts.tol should be a small positive number.');
    end
else, opts.tol = 1e-3; end

% % if the user has an initial guess, store it in this option
if isfield(opts,'init')
    if numel(opts.init) ~= 1
    end
else, opts.init = 1; end

% display options
if ~isfield(opts,'disp'), opts.disp = false; end

% Nonnegativity constraint
if isfield(opts,'nonneg')
    if ~islogical(opts.nonneg)
        error('opts.nonneg should be true or false.');
    end
else
    opts.nonneg = false;
end 

if ~isfield(opts,'isreal'), opts.isreal = true; end

% cannot recover a complex nonnegative signal
if opts.nonneg && ~opts.isreal
    opts.nonneg = false;
end

% Nesterov acceleration option
if ~isfield(opts,'fast'), opts.fast = true; end

fprintf('\nPlug-and-Play image reconstruction\n ----------------------------\n');




