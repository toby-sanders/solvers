function R = getLTBM3D(opts)

if nargin<1, opts.blockSize = 32; end

if ~isfield(opts,'blockSize'), opts.blockSize = 32; end
if ~isfield(opts,'numMax'), opts.numMax = 16; end
if ~isfield(opts,'numMin'), opts.numMin = 16; end
if ~isfield(opts,'wname'), opts.wname = 'sym'; end
if ~isfield(opts,'wnamez'), opts.wnamez = 'db'; end
if ~isfield(opts,'order'), opts.order = 4; end
if ~isfield(opts,'orderz'), opts.orderz = 1; end
if ~isfield(opts,'levels'), opts.levels = 3; end
if ~isfield(opts,'cycleSpin'), opts.cycleSpin = 2; end
if ~isfield(opts,'matchSpins'), opts.matchSpins = [0,opts.blockSize/2]; end
if ~isfield(opts,'tauMode'), opts.tauMode = 2; end
if ~isfield(opts,'filtType'), opts.filtType = 'ht'; end
if ~isfield(opts,'blockSizeWie'), opts.blockSizeWie = 16; end
if ~isfield(opts,'matchSize'), opts.matchSize = opts.blockSize; end
if ~isfield(opts,'Wiener'), opts.Wiener = true; end

R = @(z,sigma)LTBM3D(z,sigma,opts);