function final_disp(out,opts)

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% 01/09/2018
if out.total_iter == opts.iter
    fprintf('Final convergence tolerance: %10.3e\n',out.rel_chg_out(end));
    fprintf('Reached maximum number of iterations, %i\n',out.total_iter);
else
    fprintf('convergence tolerance met: %10.3e\n',out.rel_chg_out(end));
    fprintf('Number of total iterations is %i. \n',out.total_iter);
end
fprintf('||Au-b||/||b||: %5.3f\n',out.final_error);
%fprintf('Final ||w||_1: %5.3g\n',out.final_wl1);
fprintf('||Du-w||^2: %5.3g\n',out.final_Du_w);
fprintf('mu/2*||Au-b||^2 + ||Du||_1: %g\n',out.objf_val(end));
if opts.disp_fig
    if opts.disp
        figure(131);
        subplot(2,2,1);
        semilogy(out.lam1,'Linewidth',2);  %plot lam1, ||W||_1
        hold on;
        semilogy(out.lam3.*out.mu(1),'Linewidth',2);  %plot lam3, mu||Au -f||^2
        % semilogy(2:opts.inner_iter:max(size(out.f)),...
        %     out.f(2:opts.inner_iter:end),'kx','Linewidth',2);
        semilogy(out.objf_val,'linewidth',2);
        legend({'||W||_1','mu||Au - f||_2^2','obj func.'},...
            'Location','eastoutside');
        xlabel('iteration');
        grid on;
        set(gca,'fontweight','bold');
        hold off;

        subplot(2,2,2);semilogy(abs(out.optimallity2),'k*');
        title({'\textbf{optimality condition for }$\mathbf u$:';...
            '$\mu  A^H (Au-b) + \beta T^H(Tu-w)-T^H\sigma = 0$'},...
            'interpreter','latex');
        xlabel('$i^{th}$ term in optimality equality','interpreter','latex');
        legend('recovered','location','southwest');
        set(gca,'fontweight','bold','fontsize',14);
        subplot(2,2,3);
        semilogy(out.rel_error,'Linewidth',2);
        hold on
        semilogy(out.rel_chg_inn,'Linewidth',2);
        %plot(out.rel_lam2,'Linewidth',2);
        plot(2:opts.inner_iter:max(size(out.f)),...
            0,'kx','Linewidth',2);
        legend({'Rel error','Rel chg','mlp update'},'Location','eastoutside');
        xlabel('iteration');
        grid on;
        set(gca,'fontweight','bold');
        hold off
        subplot(2,2,4);hold off;
        ddd = 1e3;
        opline = linspace(-2,2,ddd)';
        opline2 = zeros(ddd,1);
        opline2(1:ddd/2) = 1;
        opline2(ddd/2+1:end) = -1;
        plot(out.W(:),out.optimallity(:),'k*');hold on;
        plot(opline,opline2,'g--','linewidth',2);
        title({'\textbf{optimality condition for }$\mathbf w$:';...
            '$\beta(w-Tu) + \sigma\in -sign(w) $'},...
            'interpreter','latex');
        xlabel('$\mathbf{w_i}$','interpreter','latex');
        ylabel('$\mathbf{\beta (w - Du)_i + \sigma_i}$','interpreter','latex');
        set(gca,'fontweight','bold','fontsize',14);
        axis([-.1,.1,-1.1,1.1]);
        legend('recovered','feasibility line');
        hold off;

    end
end