%% parameters
clc
clear
p_N = 500;
p_batch = 5; 
gpuDevice(3);
tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
idx_i = [1:1:p_N];
idx_j = ceil(idx_i/p_batch);
M_trans = gpuArray(sparse(idx_i,idx_j,1));
global dim; dim =4;
lambda = 0.1; gama = 0.5; sigma =0.1; alpha = 50;
t_N = 100000; t_now= 0;
sigma0 = 0.1; 
decay = 0.99;
%% parameters for DNN
global Layer; Layer = [dim,40,40,1];                                

N_W = 0;
N_b = 0;
for i = 1:size(Layer,2)-1
    N_W = N_W + Layer(i)*Layer(i+1);
    N_b = N_b + Layer(i+1);
end
theta_W_record = 2*rand(p_N,N_W,'single','gpuArray')-1;
theta_b_record = 2*rand(p_N,N_b,'single','gpuArray')-1;

theta_record = [theta_b_record theta_W_record];
theta_W_Num = size(theta_W_record,2);
theta_b_Num = size(theta_b_record,2);
theta_Num = size(theta_record,2);
theta_batch = 100;

%% Main part
f = zeros(p_N,1,'single','gpuArray');

                                            
loss_record = zeros(t_N,1,'single');
error_record_2 = zeros(t_N,1,'single');
error_record_inf = zeros(t_N,1,'single');
loss_record(1,1) = gather(max(fcn_DRM(theta_W_record,theta_b_record)));
error_record_2(1,1) = gather(min(fcn_error_2(theta_W_record,theta_b_record)));
error_record_inf(1,1)= gather(min(fcn_error_inf(theta_W_record,theta_b_record)));
m = zeros(size(theta_record),'single','gpuArray');
v = zeros(size(theta_record),'single','gpuArray');
diff = ones(size(theta_record),'single','gpuArray') ; 

theta_rp = randperm(theta_Num);
t = 0;
beta1 = 0.9;
beta2 = 0.99;
%while max(max(abs(diff)))>10^(-5)

while t<100000
    %lambda = lambda1/(1+exp(t/1000));
    t = t+1;
    if t == 15000
        p_batch = 10;
        lambda = 0.08;
        tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
        idx_i = [1:1:p_N];
        idx_j = ceil(idx_i/p_batch);
        M_trans = gpuArray(sparse(idx_i,idx_j,1));
    end
    if t == 20000
        p_batch = 20;
        lambda = 0.064;
        tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
        idx_i = [1:1:p_N];
        idx_j = ceil(idx_i/p_batch);
        M_trans = gpuArray(sparse(idx_i,idx_j,1));
    end
    if t== 30000
        p_batch = 20;
        lambda = 0.05
        tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
        idx_i = [1:1:p_N];
        idx_j = ceil(idx_i/p_batch);
        M_trans = gpuArray(sparse(idx_i,idx_j,1));
    end
    if t== 40000
        p_batch = 25;
        lambda = 0.04
        tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
        idx_i = [1:1:p_N];
        idx_j = ceil(idx_i/p_batch);
        M_trans = gpuArray(sparse(idx_i,idx_j,1));
    end
    if t== 45000
        p_batch = 25;
        lambda = 0.02;
        tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
        idx_i = [1:1:p_N];
        idx_j = ceil(idx_i/p_batch);
        M_trans = gpuArray(sparse(idx_i,idx_j,1));
    end
    p_rp = randperm(p_N);
    f = fcn_DRM(theta_W_record,theta_b_record);
    loss_record(t,1) = gather(max(f));
    error_record_2(t,1) = gather(min(fcn_error_2(theta_W_record,theta_b_record)));
    error_record_inf(t,1) = gather(min(fcn_error_inf(theta_W_record,theta_b_record)));
    idx = reshape(p_rp,p_batch,[]);
    f_batch = f(idx);
    f_min = min(f_batch);
    omega_temp = exp(-alpha*(f_batch-f_min));
    omega = sparse(tmp_index,idx,double(omega_temp));
    sum_omega = sum(omega_temp);
    aver_temp = omega*double(theta_record)./double(sum_omega');
    aver(p_rp,:) = single(M_trans*aver_temp);
    diff = theta_record - aver;
    diff(isnan(diff)) =0;
    m = beta1*m+ (1-beta1)*diff ;
    v = beta2*v + (1-beta2)*diff.^2;
    m_bias = m/(1-beta1^t);
    v_bias = v/(1-beta2^t);
    if t< 30000
        Gradient = - lambda * m_bias./(sqrt(v_bias) + 1e-8)+  decay^(t/10)*(rand(size(m_bias),'gpuArray')-0.5);
    else
        Gradient = - lambda * m_bias./(sqrt(v_bias) + 1e-8);
    end
    theta_record = theta_record +Gradient;
    theta_b_record = theta_record(:,1:theta_b_Num);
    theta_W_record = theta_record(:,theta_b_Num+1:end);
    
    if mod(t,50)==0
        lambda
        x = rand(201,dim);
        x(:,1) = [-1:0.01:1]';
        x(:,2:4) = 0;
        [Value,Index] = min(error_record_inf(t,:));
        theta_W = theta_W_record(Index,:);
        theta_b = theta_b_record(Index,:);
        y_true=sol_exact(x);
        y_pred=gather(DNN(x,theta_W,theta_b));
        save('approximatefunction')
        %sprintf('diff: %f',max(max(abs(diff))))
        %x = error_record_infty;
        %figure(1)
        %plot(log10(x))
        %sprintf('epoch =%d,error= %f',t ,min(error_record_infty(t)))
        %savefig("error_inf.fig")
        %x = error_record_2;
        %figure(1)
        %plot(log10(x))
        %sprintf('epoch =%d,error= %f',t ,min(error_record_2(t)))
        %savefig("error_2.fig")
        %figure(2)
        %theta_W = gather(theta_W_record(1,:));
        %theta_b = gather(theta_b_record(1,:));
        %Z=reshape(DNN(input,theta_W,theta_b),[21,21]);
        %surf(X,X',Z);
        %figure(3)
        %x = loss_record;
        %plot(log10(x))
        %sprintf('epoch =%d,error= %f',t ,min(loss_record(t)))
        %x = -1:0.1:1;
        %savefig("loss.fig")
    end
end