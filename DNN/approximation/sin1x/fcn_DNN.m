function x = fcn_DNN(x,theta_W_record,theta_b_record)
%FDN_DNN Summary of this function goes here
%   Detailed explanation goes here
global Layer;
%multi = x.^2-1;
p_N = size(theta_W_record,1);
index_W = 1;
index_b = 1;
Batch_size = size(x,1);
dim = size(x,2);
x = reshape(x',1,1,dim,Batch_size);
for i = 1:size(Layer,2)-2
    theta_W_tem = reshape(theta_W_record(:,index_W:index_W+Layer(i)*Layer(i+1)-1),p_N,Layer(i+1),Layer(i));
    theta_b_tem = reshape(theta_b_record(:,index_b:index_b+Layer(i+1)-1),p_N,Layer(i+1),1);
    x = permute(bsxfun(@plus, theta_b_tem, sum(bsxfun(@times, theta_W_tem,x),3)),[1,3,2,4]);
    x = activation2(x);
    index_W = index_W + Layer(i)*Layer(i+1);
    index_b = index_b + Layer(i+1);
end
i = size(Layer,2)-1;
theta_W_tem = reshape(theta_W_record(:,index_W:index_W+Layer(i)*Layer(i+1)-1),p_N,Layer(i+1),Layer(i));   
theta_b_tem = reshape(theta_b_record(:,index_b:index_b+Layer(i+1)-1),p_N,Layer(i+1),1);
x = permute(bsxfun(@plus, theta_b_tem, sum(bsxfun(@times, theta_W_tem,x),3)),[4,1,2,3]);
%x = multi.*x+1;
end

