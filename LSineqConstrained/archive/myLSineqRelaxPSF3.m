function [x,out] = myLSineq(A,b,xx,zz,opts)

% relaxation PSF estimation, which 
% currently solving the following minimization
% min 0.5 ||Ax-b||^2 s.t. Ex = 0, and Cx >= 0
% using a standard Lagrange multiplier method


if ~isfield(opts,'iter'), opts.iter = 100; end
if ~isfield(opts,'tol'), opts.tol = 1e-5; end
if ~isfield(opts,'LMupdate'), opts.LMupdate = 2; end
if ~isfield(opts,'gam'), opts.gam = 1e-2; end

if ~isa(A,'function_handle'), A = @(u,mode) f_handleA(A,u,mode); end
% C = getRelaxIneqOpers_local(x,z);



% set up all of the things for the equality and inequality operators
d1 = numel(xx);
d2 = numel(zz);
Sz = find(zz>=0); % points where PSF derivative is negative
Sz2 = find(zz<0); % points where PSF is zero
Sx = find(xx>=0); % points where PSF derivative (x) is negative
Sx0 = Sx;
Sx2 = zeros(numel(Sx),1); % points matching Sx2 to make PSF symmetric about x=0
for i = 1:numel(Sx)
    [~,indNew] = min(abs(-xx - xx(Sx(i))));
    Sx2(i) = indNew;
    if indNew==Sx(i)
        indDel = i;
    end
end
Sx(indDel) = ''; % delete the zero point
Sx2(indDel) = '';

% inequality (C) and equality (E) operators
C = @(U,mode)CombinedIneq(U,mode,Sx0,Sz,d1,d2);
E = @(U,mode)CombinedEqOper(U,mode,Sz2,Sx,Sx2,d1,d2);

% flg1 = check_D_Dt(@(U)E(U,1),@(U)E(U,2),[d1,d2])
% flg2 = check_D_Dt(@(U)C(U,1),@(U)C(U,2),[d1,d2])



gam = opts.gam; % step size for LM update

Atb = A(b,2);
x = zeros(size(Atb));
lambda = E(x,1);
mu = C(x,1);


out.rel_chg = [];
g2 = 0;
g3 = 0;
beta = 1e-3;
% AtA = A'*A;
for ii = 1:opts.iter
    % gradient over quadratic term   
    g1 = A(A(x,1),2) - Atb + beta*E(E(x,1),2);
    g = g1+g2+g3;
    
    % get step length, tau
    if ii~=0% ii==1 
        Ag = A(g,1);
        Eg = E(g,1);
        Cg = C(g,1);
        tau = (g'*g1 + lambda'*Eg + mu'*Cg)/(Ag'*Ag + beta*Eg'*Eg); % step length
        if isnan(tau)
            tau = 1e-5;
        end
    else % bb-step
        st = x-xp;
        yp = g-gp;
        tau =  (st'*st)/(st'*yp);
    end
    
    xp = x;
    gp = g;
    x = x - tau*g; % descent
    x = reshape(x,d1,d2);
    x(:,Sz2) = 0;
    x = max(x(:),0);

    % Lagrange multiplier update 
    % multipliers for positive inequality contraint are non-positive
    if mod(ii,opts.LMupdate)==0 
        lambda = lambda + beta*(E(x,1));% gam*(C*x-d); 
        mu = mu + gam*(C(x,1));
        mu = min(mu,0);

        g2 = E(lambda,2);% C'*lambda;   % update gradient on LM term
        g3 = C(mu,2);
    end
    out.rel_chg = [out.rel_chg;myrel(x,xp)]; 

    if ii>50 % check convergence
        if out.rel_chg(end)<opts.tol
            out.lambda = lambda;
            return;
        end
    end
end

% output the contraint values
out.lambda = lambda;
out.ineq = C(x,1);
out.ineq1 = out.ineq(1:d1*d2);
cnt = d1*d2;
out.ineq2 = reshape(out.ineq(cnt+1:cnt+(numel(Sz)-1)*d1),d1,numel(Sz)-1);
cnt = cnt+(numel(Sz)-1)*d1;
out.ineq3 = reshape(out.ineq(cnt+1:cnt+(numel(Sx)-1)*numel(Sz)),numel(Sx)-1,numel(Sz));
out.eqVals = E(x,1);

























function y = CombinedIneq(U,mode,Sx,Sz,d1,d2)
switch mode
    case 1 % forward
        % forward operation for the inequality contraints   
        U = reshape(U,d1,d2);
        y1 = U(:); % nonnegativity term (identity map)
        
        % negative derivative along the z-axis for values z>=0
        y2 = -(U(:,Sz(1)+1:end)-U(:,Sz(1):end-1));
    
        % negative derivative along the x-axis for values x>=0, z>=0
        y3 = -(U(Sx(1)+1:end,Sz)-U(Sx(1):end-1,Sz));
        
        % concatonate the three terms into single vector
        y = cat(1,y1,y2(:),y3(:));
    case 2 % adjoint

        y1 = U(1:d1*d2); % nonnegativity term (indentity map)

        % initialize last two terms. Add a small buffer to simplify the
        % derivative term. The extra dimension will vanish after taking derv.
        y2 = zeros(d1,d2+1);
        y3 = zeros(d1+1,d2);
    
        % populate y2 with appropriate values in y, then take derivative along
        % the z-axis
        cnt = d1*d2;
        y2(:,Sz(2:end)) = reshape(U(cnt+1:cnt+(numel(Sz)-1)*d1),d1,numel(Sz)-1);
        y2 = y2(:,2:end) - y2(:,1:end-1);
        
        % same as y2, now along the x-axis and only for x>=0, z>=0
        cnt = cnt+(numel(Sz)-1)*d1;
        y3(Sx(2:end),Sz) = reshape(U(cnt+1:cnt+(numel(Sx)-1)*numel(Sz)),numel(Sx)-1,numel(Sz));
        y3 = y3(2:end,:) - y3(1:end-1,:);
        
        y = y1(:) + y2(:) + y3(:);
end




function y = CombinedEqOper(U,mode,Sz,Sx,Sx2,d1,d2)

switch mode

    case 1
        U = reshape(U,d1,d2);
        y1 = U(:,Sz);
    
        y2 = U(Sx,:) - U(Sx2,:);
    
        y = cat(1,y1(:),y2(:));

    case 2
        y1 = reshape(U(1:d1*numel(Sz)),d1,numel(Sz));
        y2 = reshape(U(d1*numel(Sz)+1:end),numel(Sx),d2);
    
        U1 = zeros(d1,d2);U2 = U1;
        U1(:,Sz) = y1;
    
        U2(Sx2,:) = -y2;
        U2(Sx,:) = y2;
    
        y = U1 + U2;
        y = y(:);

end


