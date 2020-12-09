load('approximatefunction.mat')
[vlaue,i]=min(error_temp_infty);
theta_W = theta_W_record(i,:);
theta_b = theta_b_record(i,:);
x = [-1:0.1:1]';
X = meshgrid(x);
input = zeros(length(X)^2,2);
input(:,1) = reshape(X, ([length(X)^2,1]));
input(:,2) = reshape(X',([length(X)^2,1]));
figure(1);
Z1=reshape(DNN(input,theta_W,theta_b),[21,21]);
surf(X,X',Z1);
figure(2);
Z2=reshape(sol_exact(input),[21,21]);
surf(X,X',Z2);
figure(3)
surf(X,X',Z2-Z1)
print CBO_ADAM_14_error_2D.eps -depsc2 -r600
figure(4)
plot(log10(error_record_2))
print CBO_ADAM_14_process_2D_error2.eps -depsc2 -r600
figure(5)
plot(log10(error_record_infty))
print CBO_ADAM_14_process_2D_errorinfty.eps -depsc2 -r600
