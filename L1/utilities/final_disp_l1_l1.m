function final_disp_l1_l1(out,opts,fnum)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 09/18/2017

if nargin<3
    fnum = 55;
end

if out.total_iter == opts.iter
    fprintf('Final convergence tolerance: %10.3e\n',out.rel_chg_out(end));
    fprintf('Reached maximum number of iterations, %i\n',out.total_iter);
else
    fprintf('convergence tolerance met: %10.3e\n',out.rel_chg_out(end));
    fprintf('Number of total iterations is %i. \n',out.total_iter);
end
% fprintf('||Au-b||/||b||: %5.3f\n',out.final_error);
% %fprintf('Final ||w||_1: %5.3g\n',out.final_wl1);
% fprintf('||Du-w||^2: %5.3g\n',out.final_Du_w);
% fprintf('Number of total iterations is %d. \n',out.total_iter);
% fprintf('final obj func value: %5.3f\n',out.final_error);
% fprintf('Final ||w||_1: %5.3g\n',out.final_wl1);
% fprintf('Final ||Du-w||^2: %5.3g\n',out.final_Du_w);
if opts.disp_fig
    if opts.disp
        figure(fnum);
        subplot(2,1,1);
        plot(out.lam1,'Linewidth',2);  %plot lam1, || ||_2^2
        hold on;
        plot(out.lam2,'Linewidth',2);  %plot lam2, ||w||_1
        plot(abs(out.f),'Linewidth',2);   %plot f, the objective function
        plot(1:opts.inner_iter:numel(out.f),...
            out.f(1:opts.inner_iter:end),'kx','Linewidth',2);
        legend({'|| ||_2^2','||W||_1',...
            'obj function','mlp update'},...
            'Location','eastoutside');
        xlabel('iteration');
        grid on;
        set(gca,'fontweight','bold');
        hold off;

        subplot(2,1,2);
        plot(out.rel_error,'Linewidth',2);
        hold on
        plot(out.rel_chg_inn,'Linewidth',2);
        plot(out.rel_lam2,'Linewidth',2);
        plot(2:opts.inner_iter:max(size(out.f)),...
            0,'kx','Linewidth',2);
        legend({'Rel error','Rel chg','Rel lam2','mlp update'},'Location','eastoutside');
        axis([0 numel(out.rel_error) 0 .2]);
        xlabel('iteration');
        grid on;
        set(gca,'fontweight','bold');
        hold off

    end
end