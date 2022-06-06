% To run all of the algorithms in this package, add the folders to the MATLAB
% paths. One should just add these paths into the MATLAB startup.m file.
mpathlight = pwd;
slashfind = strfind(mpathlight,filesep);
mpathlight = mpathlight(1:slashfind(end)-1);
% rmpath('C:\Users\toby.sanders\Documents\repos\nga-fmv\BM3D');
% rmpath([mpathlight,'']);

rmpath([mpathlight,'/MI_base']);
rmpath([mpathlight,'/MI_base/utilities']);
rmpath([mpathlight,'/MI_base/operators']);

rmpath([mpathlight,'/Deblur/PSFs']);
rmpath([mpathlight,'/Deblur/Operators']);
rmpath([mpathlight,'/Deblur/SURE']);
% rmpath([mpathlight,'/Deblur/MFBD']);

%%% No longer need these denoising engines... 
% rmpath([mpathlight,'/solvers/DenoisingEngines/BM3D']);
% rmpath([mpathlight,'/solvers/DenoisingEngines/BM3D/bm3d']);
% rmpath([mpathlight,'/solvers']);
% rmpath([mpathlight,'/solvers/DenoisingEngines']);
% rmpath([mpathlight,'/solvers/DenoisingEngines/TNRD']);
% rmpath([mpathlight,'/solvers/DenoisingEngines/GBM3D']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GBM3D functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmpath([mpathlight,'/GBM3D/waveTransforms2D']);
rmpath([mpathlight,'/GBM3D/waveTransforms3D']);
rmpath([mpathlight,'/GBM3D/utilities']);
rmpath([mpathlight,'/GBM3D/source']);
rmpath([mpathlight,'/GBM3D/MilinfarEnhance']);
% rmpath([mpathlight,'/GBM3D/waveletFilters']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plug and play prior algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmpath([mpathlight,'/PnP/source']);
rmpath([mpathlight,'/PnP/utilities']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1-L2 optimization solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmpath([mpathlight,'/solvers/L1']);
rmpath([mpathlight,'/solvers/RL']);
rmpath([mpathlight,'/solvers/L1/utilities']);
rmpath([mpathlight,'/solvers/L1/Transforms']);
rmpath([mpathlight,'/solvers/L1/Transforms/multiscale']);
rmpath([mpathlight,'/solvers/L1/Transforms/wave_shear']);
rmpath([mpathlight,'/solvers/L1/joint']);
rmpath([mpathlight,'/solvers/L1/inpaint']);
rmpath([mpathlight,'/solvers/L2']);
rmpath([mpathlight,'/solvers/tikhonov']);
rmpath([mpathlight,'/solvers/parm_selection']);
rmpath([mpathlight,'/solvers/parm_selection/Skeel']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% utilities, tomography, radar, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmpath([mpathlight,'/utilities']);
% rmpath([mpathlight,'/utilities/MRCreading']);
% rmpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10/2D'])
% rmpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10/3D'])
% rmpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10/Util'])
% rmpath([mpathlight,'/utilities/ShearLab3Dv10/ShearLab3Dv10'])

rmpath([mpathlight,'/SandBox/tomography']);
rmpath([mpathlight,'/SandBox/tomography/Align'])
rmpath([mpathlight,'/SandBox/tomography/Align/subroutines'])
rmpath([mpathlight,'/SandBox/tomography/JOHANN_RADON'])
rmpath([mpathlight,'/SandBox/tomography/DART'])
rmpath([mpathlight,'/SandBox/tomography/DART/subroutines'])


rmpath([mpathlight,'/SandBox/tomography/DIPS'])
rmpath([mpathlight,'/SandBox/tomography/DIPS/solver'])
rmpath([mpathlight,'/SandBox/tomography/SIRT']);


rmpath([mpathlight,'/SandBox/radar']);
rmpath([mpathlight,'/SandBox/radar/raider-code']);
rmpath([mpathlight,'/SandBox/radar/SAR_setup']);
rmpath([mpathlight,'/SandBox/radar/SAR_setup/utilities']);

%add Fessler's paths
irtdir = [mpathlight,'/SandBox/fast-transforms/fessler/'];
rmpath([irtdir 'nufft/greengard']);
rmpath([irtdir 'align']);		% image registration
rmpath([irtdir 'align/mex']);		% image registration mex files
rmpath([irtdir 'blob']);		% blob (KB) basis
rmpath([irtdir 'ct']);			% x-ray CT (polyenergetic) recon
rmpath([irtdir 'data']);		% example data
rmpath([irtdir 'emission']);		% emission image reconstruction
rmpath([irtdir 'example']);		% example applications
rmpath([irtdir 'fbp']);		% FBP (filtered backprojection) code
rmpath([irtdir 'general']);		% generic image reconstruction
rmpath([irtdir 'graph']);		% graphics routines
rmpath([irtdir 'mri']);		% MRI reconstruction
rmpath([irtdir 'mri-rf/yip-spsp']);	% MRI RF pulse design
%rmpath([irtdir 'mri/recon']);		% MRI reconstruction - old
rmpath([irtdir 'nufft']);		% nonuniform FFT (for a fast projector)
rmpath([irtdir 'nufft/table']);	% mex files for NUFFT
rmpath([irtdir 'penalty']);		% regularizing penalty functions
rmpath([irtdir 'systems']);		% system "matrices"
rmpath([irtdir 'systems/tests']);	% tests of systems
rmpath([irtdir 'transmission']);	% transmission image reconstruction
rmpath([irtdir 'utilities']);		% various utility functions
rmpath([irtdir 'wls']);		% weighted least-squares (WLS) estimates
%re1 = @(x)reshape(x,size(x,1)*size(x,2)*size(x,3),1);
%re2 = @(x)reshape(x,sqrt(size(x,1)),sqrt(size(x,1)));
clear mainpath;
clear irtdir;
clear mpathlight;
clear slashfind;

fprintf('WELCOME! :)\n')