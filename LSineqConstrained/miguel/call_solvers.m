clear;
load lsqlin_test_2;


[x1,~,~,exitflag,~,lambda] = lsqlin(Mfit,rhsfit,Mineq,rhsineq,Meq,rhseq,[],[],[],options);
       
opts.iter = 1000;
opts.tol = 1e-8;
myC = cat(1,-Mineq,Meq,-Meq);
myD = cat(1, -rhsineq,rhseq,-rhseq);
cd ../
[x2,out2] = myLSineqExact_preCond(Mfit,rhsfit,myC,myD,opts);
[x3,out3] = myLSineqExact_Msplit(Mfit,rhsfit,myC,myD,opts);
% [x4,out4] = myLSineq(Mfit,rhsfit,myC,myD,opts);
norm(Mfit*x1-rhsfit)
norm(Mfit*x2-rhsfit)

norm(min(myC*x2-myD,0))
norm(min(myC*x1-myD,0))

figure(213);
subplot(2,2,1);hold off;
semilogy(out2.rel_chg);hold on;
semilogy(out3.rel_chg);
% semilogy(out4.rel_chg);hold off;
subplot(2,2,2);hold off;
plot(x2,'o');hold on;
plot(x3,'*');
plot(x1,'x');
% plot(x4,'o');
hold off;
legend({'precond diag','precond inv app','lsqlin'});
subplot(2,2,3);hold off;
plot(-out2.mu);hold on;
plot(lambda.ineqlin);
subplot(2,2,4);hold off;
plot(myC*x2-myD);hold on;
plot(myC*x1-myD);
hold off;
% plot(-out4.lambda);