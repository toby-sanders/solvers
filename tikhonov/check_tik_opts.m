function opts = check_tik_opts(opts)

% This function checks the options set up for the Tikhonov optimization code,
% HOTV3D.  Improperly labeled field options will be identified, and all
% unlabeled fields are set to default values
%
%
% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 10/08/2017

% order of the finite difference operator
% Set to 1 for TV regularization
% Set to 0 for signal sparsity
% Set to >= 2 for higher order methods
% Noninteger values are also accepted for fractional finite difference
% scheme
wflg = zeros(11,1);
if ~isfield(opts,'order')
    fprintf('Order of finite difference set to 1 (TV regularization)\n\n');
    opts.order=1;
elseif opts.order<0
    error('opts.order should be at least 0');
end

% number of decent iterations
if isfield(opts,'iter')
    if ~isscalar(opts.iter) || opts.iter <= 0
        error('opts.iter should be a positive integer.');
    end
else
    opts.iter = 200;
end

% Display option
if ~isfield(opts,'disp')
    opts.disp = true;
end

% mu is generally the most important parameter
% mu is mainly decided by noise level. Set mu big when b is noise-free
% whereas set mu small when b is very noisy.
if isfield(opts,'mu')
    if ~isscalar(opts.mu) || opts.mu <0
        error('opts.mu must be positive.');
    end
else
    %default mu
    opts.mu = 1;
end





% outer loop convergence tolerance
if isfield(opts,'tol')
    if ~isscalar(opts.tol) || opts.tol <= 0
        error('opts.tol should be a positive small number.');
    end
else
    opts.tol = 1.e-4;
end






% if the user has an initial guess, store it in this option
if isfield(opts,'init')
    if numel(opts.init) > 1
        fprintf('User has supplied opts.init as initial guess solution!!!\n');
    end
else
    opts.init = []; % initialized with single step gradient decent least squares with exact step length
end





% scaling of the operator A and vector b. Scaling HIGHLY recommended so that
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
        error('maximum constraint (max_c) was set to true without specifying the maximum value (max_v)');
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


opts.reweighted_TV = false;

%{
if wflg1, fprintf('opts.mu is not within optimal range\n'); end;
if wflg2, fprintf('opts.beta is not within optimal range\n'); end;
if wflg3, fprintf('Using periodic regularization\n'); end;
if wflg4, fprintf('opts.min_l2_error is large'); end;
if wflg5, fprintf('scale_A set to false, this is not recommended'); end;
if wflg6, fprintf('scale_b set to false, this is not recommended'); end;
if wflg(7), warning('Lagrangian multiplier is not being used for data
enforcement'); end;
if wflg(8),         warning('signal is assumed to be complex with a smooth
phase'); end;
wflg(9), warning('signal set to be real but smooth_phase set false.');

wflg(10), warning('correction phase angles not specified, using Atb');
%}

msgs = ...
    {'-opts.mu may not be within optimal range';
    '-opts.beta may not be within optimal range';
    '-Using periodic regularization';
    '-opts.min_l2_error is large';
    '-scale_A set to false (not recommended)';
    '-scale_b set to false (not recommended)';
    '-Lagrangian multiplier is being used to encourage the constrained problem';
    '-signal is assumed to be complex with a smoothly varying phase';
    '-signal set to be real but smooth_phase set false'
    '-estimate phase angles not specified, using Atb'
    '-opts.nonneg reset to false since opts.isreal is false'};



    
%     
% fprintf('\nTikhonov, order %g, %g level(s)\n-----------------------------\n'...
%     ,opts.order,opts.levels);
% % 
% % fprintf('***********************************')
% % fprintf('\n*    HOTV %g L2 regularization        *\n',opts.order);
% % fprintf('*          %g level(s)             *\n',opts.levels);
% % fprintf('***********************************\n')
% % 
% 
% if sum(wflg)
%     fprintf('WARNINGS:\n');
%     disp(msgs(find(wflg)));
% end
%     






