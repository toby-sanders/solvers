function R = getTNRD(scl)

load('JointTraining_7x7_400_180x180_stage=5_sigma=5.mat','MFS','cof','stage');
if nargin<1, scl = []; end
R = @(z)myTNRDcaller(z,MFS,cof,stage,scl);

function z = myTNRDcaller(z,MFS,cof,stage,scl)

if isempty(scl), scl = max(z(:));  end
z = ReactionDiffusion(z*255/scl,MFS,cof,stage);
z = z*scl/255;