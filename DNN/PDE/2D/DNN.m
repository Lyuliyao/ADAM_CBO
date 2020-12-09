function x = DNN(x,theta_W,theta_b)
% x = rand(1000,2);
% theta_W = rand(m,n,n);
% theta_b = rand(m,n)
% m depth
% n width
% d dimension
global Layer;
index_W = 1;
index_b = 1;
for i = 1:size(Layer,2)-2
    theta_W_tem = reshape(theta_W(index_W:index_W+Layer(i)*Layer(i+1)-1),Layer(i+1),Layer(i));
    theta_b_tem = reshape(theta_b(index_b:index_b+Layer(i+1)-1),Layer(i+1),1);
    x = (theta_W_tem*x' + theta_b_tem)';
    x = activation(x);
    index_W = index_W + Layer(i)*Layer(i+1);
    index_b = index_b + Layer(i+1);
end
i = size(Layer,2)-1;
theta_W_tem = reshape(theta_W(index_W:index_W+Layer(i)*Layer(i+1)-1),Layer(i+1),Layer(i));
theta_b_tem = reshape(theta_b(index_b:index_b+Layer(i+1)-1),Layer(i+1),1);
x = (theta_W_tem*x' + theta_b_tem)';
end


