function opts = check_HOTV_opts(opts)

% This function checks the options set up for the L1 optimization code,
% HOTV3D.  Improperly labeled field options will be identified, and all
% unlabeled fields are set to default values
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% last update: 01/08/2018

% order of the finite difference operator
% Set to 1 for TV regularization
% Set to 0 for signal sparsity
% Set to >= 2 for higher order methods
% Noninteger values are also accepted for fractional finite differences
wflg = zeros(12,1);
if ~isfield(opts,'order')
    fprintf('Order of finite difference set to 1 (TV regularization)\n\n');
    opts.order=1;
elseif opts.order<0, error('opts.order should be at least 0'); end

% can simply specify iter to avoid selecting inner and outer iterations
if isfield(opts,'iter')
    if ~isscalar(opts.iter) || opts.iter <= 0
        error('opts.iter should be a positive integer.');
    end
else, opts.iter = 250; end

% number of inner loop iterations
% inner_iter gives the number of iterations (alternating gradient decent 
% with shrinkage) for each set of Lagrangian multipliers
if isfield(opts,'inner_iter')
    if ~isscalar(opts.inner_iter) || opts.inner_iter <= 0
        error('opts.inner_iter should be a positive integer.');
    end
else, opts.inner_iter = 10; end

% The user has the option to use Lagrangian multipliers for the data term.
% Default is to not use the Lagrangian multipliers.  With a sufficient number
% of outer iterations, this option set to true will approximately solve
% Au=b.
% There is no option for this for the sparsity term.  The sparsity 
% multipliers should be used to enforce the constrained problem, Du = w.
if ~isfield(opts,'data_mlp'), opts.data_mlp = false;
elseif opts.data_mlp, wflg(7) = 1; end
if islogical(opts.data_mlp) & opts.data_mlp
    opts.data_mlp = opts.iter;
end

if islogical(opts.data_mlp) & opts.data_mlp
    opts.data_mlp = opts.iter;
end

if ~isfield(opts,'dataLM')
    opts.dataLM = false;
end

if ~isfield(opts,'mode')
    opts.mode = 'GD';
end

if strcmp(opts.mode,'deconv')
    if ~isfield(opts,'automateMu'), opts.automateMu = true; end
end
% mu is generally the most important parameter
% mu is mainly decided by noise level. Set mu big when b is noise-free
% whereas set mu small when b is very noisy.
if isfield(opts,'mu')
    if ~isscalar(opts.mu) || opts.mu <0
        error('opts.mu must be positive.');
    end
else, opts.mu = 90; end

% coefficient for sparsifying operator
% setting beta = 2^5 usually works well
% beta is rescaled later however, depending on data vector b....
if isfield(opts,'beta')
    if ~isscalar(opts.beta) || opts.beta <0
        error('opts.beta must be positive.');
    end
else, opts.beta = 2^5; end


% option for periodic regularization
% Typically we do not want to apply the shrinkage at the boundaries
% In this case set wrap_shrink to false
if ~isfield(opts,'wrap_shrink'), opts.wrap_shrink = false;
elseif opts.wrap_shrink, wflg(3) = 1; end




% continuation parameter for mu and beta
% after each outer iteration, mu = min(muf , mu*opts.rate_ctn)
if isfield(opts,'rate_ctn')
    if ~isscalar(opts.rate_ctn) %|| opts.rate_ctn <= 1
        error('opts.rate_ctn is either not a scalar or no bigger than one.');
    end
else, opts.rate_ctn = 1.5; end







% Convergence based on the relative l2 error
if isfield(opts,'min_l2_error')
    if opts.min_l2_error<0 || opts.min_l2_error>=1
        error('opts.min_l2_error should be between 0 and 1');
    elseif opts.min_l2_error>.2
        wflg(4) = 1;
    end
else, opts.min_l2_error=0; end


% convergence tolerance
if isfield(opts,'tol')
    if ~isscalar(opts.tol) || opts.tol<=0 || opts.tol>.1
        error('opts.tol should be a small positive number.');
    end
else, opts.tol = 1e-4; end

% if the user has an initial guess, store it in this option
if isfield(opts,'init')
    if numel(opts.init) ~= 1
        wflg(12)=1;
    end
else, opts.init = 1; end

% display options
if ~isfield(opts,'disp'), opts.disp = false; end
if ~isfield(opts,'disp_fig'), opts.disp_fig = opts.disp; end

% scaling of the operator A. Scaling HIGHLY recommended so that
% consistent values for mu and beta may be used independent of the problem
% A is scaled so that ||A||_2 = 1.
if isfield(opts,'scale_A')
    if ~islogical(opts.scale_A)
        error('opts.scale_A should be true or false.');
    end
    if ~opts.scale_A
        wflg(5) = 1;        
    end
else
    opts.scale_A = true;
end

if isfield(opts,'scale_mu')
    if ~islogical(opts.scale_mu)
        error('opts.scale_mu should be true or false.');
    end
    if ~opts.scale_mu
        wflg(6) = 1;
    end
    
else
    opts.scale_mu = true;
end

% option to store the solution at each iterate
% generally this should be false since it may require significant memory
if isfield(opts,'store_soln')
    if ~islogical(opts.store_soln)
        error('opts.store_soln should be true or false.');
    end
else
    opts.store_soln = false;
end

% Nonnegativity constraint
if isfield(opts,'nonneg')
    if ~islogical(opts.nonneg)
        error('opts.nonneg should be true or false.');
    end
else
    opts.nonneg = false;
end 


%if the data is complex, such as fourier data, but the signal is real, set
%this option to true to search for a real solution
if isfield(opts,'isreal')
    if ~islogical(opts.isreal)
        error('opts.isreal should be true or false.');
    end
    % if the signal is not necessarily real, its absolute value may be smooth
    % but the phase at each pixel may be totally random.  In this case, set 
    % smooth_phase to false, and specify the estimated phase angle (in radians) 
    % at each pixel into opt.phase_angles
    if ~isfield(opts,'smooth_phase')
        opts.smooth_phase = true;
    elseif ~opts.smooth_phase
        if opts.isreal
            wflg(9) = 1;       
            %opts.smooth_phase = true;
        end
    end
    if ~opts.isreal && opts.smooth_phase
        wflg(8) = 1;
    end
else
        opts.isreal = true;
        opts.smooth_phase = true;
end

% Maximum value constraint
if ~isfield(opts,'max_c')
    opts.max_c = false;
elseif opts.max_c
    if ~isfield(opts,'max_v')
        error(['maximum constraint (max_c) was set to true without',...
            ' specifying the maximum value (max_v)']);
    end
end

% cannot recover a complex nonnegative signal
if opts.nonneg && ~opts.isreal
    opts.nonneg = false;
    wflg(11) = 1;
end

% cannot enforce maximum value constraint on complex signal
if opts.max_c && ~opts.isreal
    opts.max_c = false;
    warning('opts.max_c reset to false, since opts.isreal is false');
end

% number of scalings in the finite difference operator
if ~isfield(opts,'levels')
    opts.levels = 1;
elseif round(opts.levels)~=opts.levels || opts.levels < 1
    error('opts.levels should be a positive integer');
end

% if adaptive is true, then the method is using reweighted FD transform
% the new weights should be put into opts.coef, which is a 3x1 cell
if ~isfield(opts,'reweighted_TV')
    opts.reweighted_TV=false;
elseif opts.reweighted_TV == true && ~isfield(opts,'coef')
    error('Reweighted norm was set true without specifying the weights');
end

if ~isfield(opts,'phase_angles')
    opts.phase_angles = false;
end

if ~opts.smooth_phase && sum(sum(sum(opts.phase_angles)))==0
    wflg(10) = 1;    
end


if ~isfield(opts,'update_phase')
    if ~opts.smooth_phase
        opts.update_phase = true;
    else
        opts.update_phase = false;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The remaining parameters are for the gradient decent and backtracking %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Defaults for these parameters is recommended

% gamma is for backtracking
if isfield(opts,'gamma')
    if ~isscalar(opts.gamma) || opts.gamma <= 0 || opts.gamma > 1
        error('opts.gamma should be a scalar between 0 and 1.');
    end
else
    opts.gamma = .6;
end



% Control the degree of nonmonotonicity. 0 corresponds to monotone line search.
% The best convergence is obtained by using values closer to 1 when the iterates
% are far from the optimum, and using values closer to 0 when near an optimum.
if isfield(opts,'gam')
    if ~isscalar(opts.gam) || opts.gam <= 0 || opts.gam > 1
        error('opts.gam should be a scalar between 0 and 1.');
    end
else
    opts.gam = .9995;
end



% shrinkage rate of gam
if isfield(opts,'rate_gam')
    if ~isscalar(opts.rate_gam) || opts.rate_gam <= 0 || opts.rate_gam > 1
        error('opts.rate_gam should be a scalar between 0 and 1.');
    end
else
    opts.rate_gam = .9;
end



% tau is the step length in the gradient decent
if isfield(opts,'tau')
    if ~isscalar(opts.tau) || opts.tau <= 0
        error('opts.tau is not positive scalar.');
    end
else
    opts.tau = 1.8;
end

% L1type either isotropic or anisotropic
if ~isfield(opts,'L1type')
    if opts.levels>1 || opts.order==0
        opts.L1type = 'anisotropic';
    else
        opts.L1type = 'isotropic';
    end
elseif ~sum(strcmp(opts.L1type,{'isotropic','anisotropic'}))
    warning('L1type not recognized, set to default');
    if opts.levels>1 || opts.order==0
        opts.L1type = 'anisotropic';
    else
        opts.L1type = 'isotropic';
    end
end

if opts.levels>1 || opts.order==0
        opts.L1type = 'anisotropic';
end


if ~isfield(opts,'PSD')
    opts.PSD = 0;
elseif sum(abs(opts.PSD(:)))
    opts.L1type = 'anisotropic';
    opts.wrap_shrink = true;
end

    



msgs = ...
{'opts.mu may not be within optimal range';
'opts.beta may not be within optimal range';
'Using periodic regularization';
'opts.min_l2_error is large';
'scale_A set to false';
'scale_mu set to false';
'Lagrangian multiplier is being used to encourage the constrained problem';
'signal is assumed to be complex with a smoothly varying phase';
'signal set to be real but smooth_phase set false'
'estimate phase angles not specified, using Atb'
'opts.nonneg reset to false since opts.isreal is false';
'User has supplied opts.init as initial guess solution'};
    
if opts.disp
    fprintf('\nHOTV, order %g, %g level(s)\n--------------------------\n'...
        ,opts.order,opts.levels);
end

if ~opts.isreal && ~opts.smooth_phase
    fprintf('PHASE ESTIMATED scheme\n');
end
if opts.disp
    if sum(wflg)
        fprintf('Notes:\n');
        disp(msgs(find(wflg)));
    end
end
    






