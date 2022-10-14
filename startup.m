% To run all of the algorithms in this package, add the folders to the MATLAB
% paths. One should just add these paths into the MATLAB startup.m file.
mpathlight = pwd;
slashfind = strfind(mpathlight,filesep);
mpathlight = mpathlight(1:slashfind(end)-1);
% addpath('C:\Users\toby.sanders\Documents\repos\nga-fmv\BM3D');
% addpath([mpathlight,'']);

addpath([mpathlight,'/MI_base']);
addpath([mpathlight,'/MI_base/utilities']);
addpath([mpathlight,'/MI_base/operators']);
addpath([mpathlight,'/MI_base/functions']);
addpath([mpathlight,'/MI_base/signalOperators']);
addpath([mpathlight,'/MI_base/PSFs']);

addpath([mpathlight,'/Deblur/PSFs']);
addpath([mpathlight,'/Deblur/Operators']);
addpath([mpathlight,'/Deblur/SURE']);
% addpath([mpathlight,'/Deblur/MFBD']);

%%% No longer need these denoising engines... 
% addpath([mpathlight,'/solvers/DenoisingEngines/BM3D']);
% addpath([mpathlight,'/solvers/DenoisingEngines/BM3D/bm3d']);
% addpath([mpathlight,'/solvers']);
% addpath([mpathlight,'/solvers/DenoisingEngines']);
% addpath([mpathlight,'/solvers/DenoisingEngines/TNRD']);
% addpath([mpathlight,'/solvers/DenoisingEngines/GBM3D']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GBM3D functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/GBM3D/waveTransforms2D']);
addpath([mpathlight,'/GBM3D/waveTransforms3D']);
addpath([mpathlight,'/GBM3D/utilities']);
addpath([mpathlight,'/GBM3D/source']);
addpath([mpathlight,'/GBM3D/MilinfarEnhance']);
% addpath([mpathlight,'/GBM3D/waveletFilters']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plug and play prior algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/PnP/source']);
addpath([mpathlight,'/PnP/utilities']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1-L2 optimization solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/solvers/L1']);
addpath([mpathlight,'/solvers/RL']);
addpath([mpathlight,'/solvers/L1/utilities']);
addpath([mpathlight,'/solvers/L1/Transforms']);
addpath([mpathlight,'/solvers/L1/Transforms/multiscale']);
addpath([mpathlight,'/solvers/L1/Transforms/wave_shear']);
addpath([mpathlight,'/solvers/L1/joint']);
addpath([mpathlight,'/solvers/L1/inpaint']);
addpath([mpathlight,'/solvers/L2']);
addpath([mpathlight,'/solvers/tikhonov']);
addpath([mpathlight,'/solvers/parm_selection']);
addpath([mpathlight,'/solvers/LSineqConstrained']);
% addpath([mpathlight,'/solvers/parm_selection/Skeel']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% utilities, tomography, radar, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([mpathlight,'/utilities']);
addpath([mpathlight,'/utilities/nrrd_read_write_rensonnet']);
% addpath([mpathlight,'/utilities/MRCreading']);
% addpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10/2D'])
% addpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10/3D'])
% addpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10/Util'])
% addpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10'])

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
clear slashfind;

fprintf('WELCOME! :)\n')