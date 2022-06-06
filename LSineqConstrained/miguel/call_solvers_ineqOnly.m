clear;
load lsqlin_test_1;


[x1,~,~,exitflag,~,lambda] = lsqlin(Mfit,rhsfit,Mineq,rhsineq,[],[],[],[],[],options);
       
opts.iter = 10000;
opts.tol = 1e-8;
% myC = cat(1,-Mineq,Meq,-Meq);
% myD = cat(1, -rhsineq,rhseq,-rhseq);
cd ../
[x2,out2] = myLSineqExact(Mfit,rhsfit,-Mineq,-rhsineq,opts);



figure(213);
subplot(2,2,1);semilogy(out2.rel_chg);
subplot(2,2,2);hold off;
plot(x2,'o');hold on;
plot(x1,'x');hold off;
subplot(2,2,3,'o');hold off;
plot(-out2.mu);hold on;
plot(lambda.ineqlin);