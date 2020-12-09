load('approximatefunction.mat')
i=3;
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