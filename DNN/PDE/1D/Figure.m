load('approximatefunction.mat')
x = -1:0.1:1;
y = 0*x;
for i = 1:500
    theta_W = theta_W_record(i,:);
    theta_b = theta_b_record(i,:);
    y = y + DNN(x',theta_W,theta_b);
end
figure
Min_error = min(error_record')';
plot(log10(Min_error),'LineWidth',2)
ylabel("error",'FontSize',18)
xlabel("epoch",'FontSize',18)
set(gca,'FontSize',18,'Fontname', 'Times New Roman');
print CBO_ADAM_singular_activation_PDE_process.eps -depsc2 -r600
figure
plot(x,sol_exact(x'),'LineWidth',2)
hold on
plot(x,DNN(x',theta_W,theta_b),'LineWidth',2)
legend("Exact Solution","Numerical Solution")
set(gca,'FontSize',18,'Fontname', 'Times New Roman');
hold off
print CBO_ADAM_singular_activation_PDE_result.eps -depsc2 -r600