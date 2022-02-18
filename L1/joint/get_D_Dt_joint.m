function [D,Dt] = get_D_Dt_joint(k,p,q,r,opts)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 11/2016

if opts.smooth_phase
    if round(k) == k & opts.levels == 1
        [D,Dt] = FD2D(k,p,q,r);
    elseif opts.levels == 1
        [D,Dt] = FFD2D(k,p,q,r); 
    else
        if r~=1
            [D,Dt] = FFD3D_multiscale(k,opts.levels,p,q,r);
        else
            [D,Dt] = FFD2D_multiscale(k,opts.levels,p,q);
        end
    end
else
    if sum(sum(sum(abs(opts.phase_angles))))==0
        opts.phase_angles = exp(-1i*reshape(Atb,p,q,r));
    else
        opts.phase_angles = exp(-1i*opts.phase_angles);
    end
    if round(k) == k && opts.levels == 1
        [D,Dt] = FD2D_complex(k,p,q,r,opts.phase_angles);
    elseif opts.levels == 1
        [D,Dt] = FFD2D_complex(k,p,q,r,opts.phase_angles);
    else
        [D,Dt] = FD3D_multiscale_complex(k,opts.levels,p,q,r,opts.phase_angles); 
    end
end
%{
if isfield(opts,'orderz')
    if opts.smooth_phase
        if k~=round(k) || opts.orderz ~= round(opts.orderz) 
            [D,Dt] = FFD3D_joint(k,opts.orderz,p,q,r);
        else
            [D,Dt] = FD3D_joint(k,opts.orderz,p,q,r);
        end
    else        
        if k~=round(k) || opts.orderz ~= round(opts.orderz) 
            [D,Dt] = FFD3D_complex_joint(k,opts.orderz,p,q,r,opts.phase_angles);
        else
            [D,Dt] = FD3D_complex_joint(k,opts.orderz,p,q,r,opts.phase_angles);
        end
    end
end

% if reweighted TV, override everything else
if opts.reweighted_TV
    [D,Dt] = FD3D_weighted(k,p,q,r,opts.coef);
end
%}