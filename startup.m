% To run all of the algorithms in this package, add the folders to the MATLAB
% paths. One should just add these paths into the MATLAB startup.m file.
if exist('C:\Users\toby.sanders\Dropbox\TobySharedMATLAB','dir')
    mpathlight = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB';
elseif exist('/Users/tobysanders/Dropbox/TobySharedMATLAB','dir')
    mpathlight = '/Users/tobysanders/Dropbox/TobySharedMATLAB';
else
    mpathlight = '/home/tobysanders/Dropbox/TobySharedMATLAB';
end

% addpath('C:\Users\toby.sanders\Documents\repos\nga-fmv\BM3D');
% addpath([mpathlight,'']);


addpath([mpathlight,'/Deblur/PSFs']);
addpath([mpathlight,'/Deblur/Operators']);
addpath([mpathlight,'/Deblur/SURE']);
addpath([mpathlight,'/Deblur/MFBD']);

%%% No longer need these denoising engines... 
% addpath([mpathlight,'/TobyShared/solvers/DenoisingEngines/BM3D']);
% addpath([mpathlight,'/TobyShared/solvers/DenoisingEngines/BM3D/bm3d']);
% addpath([mpathlight,'/TobyShared/solvers']);
% addpath([mpathlight,'/TobyShared/solvers/DenoisingEngines']);
% addpath([mpathlight,'/TobyShared/solvers/DenoisingEngines/TNRD']);
% addpath([mpathlight,'/TobyShared/solvers/DenoisingEngines/LTBM3D']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LTBM3D functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/LTBM3D/waveTransforms2D']);
addpath([mpathlight,'/LTBM3D/waveTransforms3D']);
addpath([mpathlight,'/LTBM3D/utilities']);
addpath([mpathlight,'/LTBM3D/source']);
addpath([mpathlight,'/LTBM3D/MilinfarEnhance']);
% addpath([mpathlight,'/LTBM3D/waveletFilters']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plug and play prior algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/PnP/source']);
addpath([mpathlight,'/PnP/utilities']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1-L2 optimization solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/TobyShared/solvers/L1']);
addpath([mpathlight,'/TobyShared/solvers/RL']);
addpath([mpathlight,'/TobyShared/solvers/L1/utilities']);
addpath([mpathlight,'/TobyShared/solvers/L1/Transforms']);
addpath([mpathlight,'/TobyShared/solvers/L1/Transforms/multiscale']);
addpath([mpathlight,'/TobyShared/solvers/L1/Transforms/wave_shear']);
addpath([mpathlight,'/TobyShared/solvers/L1/joint']);
addpath([mpathlight,'/TobyShared/solvers/L1/inpaint']);
addpath([mpathlight,'/TobyShared/solvers/L2']);
addpath([mpathlight,'/TobyShared/solvers/tikhonov']);
addpath([mpathlight,'/TobyShared/solvers/parm_selection']);
addpath([mpathlight,'/TobyShared/solvers/parm_selection/Skeel']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% utilities, tomography, radar, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/TobyShared/utilities']);
% addpath([mpathlight,'/TobyShared/utilities/MRCreading']);
% addpath([mpathlight,'/TobyShared/utilities/ShearLab3Dv10/ShearLab3Dv10/2D'])
% addpath([mpathlight,'/TobyShared/utilities/ShearLab3Dv10/ShearLab3Dv10/3D'])
% addpath([mpathlight,'/TobyShared/utilities/ShearLab3Dv10/ShearLab3Dv10/Util'])
% addpath([mpathlight,'/TobyShared/utilities/ShearLab3Dv10/ShearLab3Dv10'])

addpath([mpathlight,'/SandBox/tomography']);
addpath([mpathlight,'/SandBox/tomography/Align'])
addpath([mpathlight,'/SandBox/tomography/Align/subroutines'])
addpath([mpathlight,'/SandBox/tomography/JOHANN_RADON'])
addpath([mpathlight,'/SandBox/tomography/DART'])
addpath([mpathlight,'/SandBox/tomography/DART/subroutines'])


addpath([mpathlight,'/SandBox/tomography/DIPS'])
addpath([mpathlight,'/SandBox/tomography/DIPS/solver'])
addpath([mpathlight,'/SandBox/tomography/SIRT']);


addpath([mpathlight,'/SandBox/radar']);
addpath([mpathlight,'/SandBox/radar/raider-code']);
addpath([mpathlight,'/SandBox/radar/SAR_setup']);
addpath([mpathlight,'/SandBox/radar/SAR_setup/utilities']);

%add Fessler's paths
irtdir = [mpathlight,'/SandBox/fast-transforms/fessler/'];
addpath([irtdir 'nufft/greengard']);
addpath([irtdir 'align']);		% image registration
addpath([irtdir 'align/mex']);		% image registration mex files
addpath([irtdir 'blob']);		% blob (KB) basis
addpath([irtdir 'ct']);			% x-ray CT (polyenergetic) recon
addpath([irtdir 'data']);		% example data
addpath([irtdir 'emission']);		% emission image reconstruction
addpath([irtdir 'example']);		% example applications
addpath([irtdir 'fbp']);		% FBP (filtered backprojection) code
addpath([irtdir 'general']);		% generic image reconstruction
addpath([irtdir 'graph']);		% graphics routines
addpath([irtdir 'mri']);		% MRI reconstruction
addpath([irtdir 'mri-rf/yip-spsp']);	% MRI RF pulse design
%addpath([irtdir 'mri/recon']);		% MRI reconstruction - old
addpath([irtdir 'nufft']);		% nonuniform FFT (for a fast projector)
addpath([irtdir 'nufft/table']);	% mex files for NUFFT
addpath([irtdir 'penalty']);		% regularizing penalty functions
addpath([irtdir 'systems']);		% system "matrices"
addpath([irtdir 'systems/tests']);	% tests of systems
addpath([irtdir 'transmission']);	% transmission image reconstruction
addpath([irtdir 'utilities']);		% various utility functions
addpath([irtdir 'wls']);		% weighted least-squares (WLS) estimates
%re1 = @(x)reshape(x,size(x,1)*size(x,2)*size(x,3),1);
%re2 = @(x)reshape(x,sqrt(size(x,1)),sqrt(size(x,1)));
clear mainpath;
clear irtdir;
clear mpathlight;

fprintf('WELCOME! :)\n')