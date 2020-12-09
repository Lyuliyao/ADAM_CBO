%% parameters
function vW_success_rate = ADAM_normal_initial_0(dim,p_N,p_batch,simu_N,decay_rate,lambda)
%p_N = 10000
%p_batch = 100

conv_success_rate = 0;
vW_success_rate = 0;
for simu = 1:simu_N

%lambda = 0.1;
alpha = 50;
%gama = 0.5; sigma =0.01;
t_N = 50001; %t_now= 0;
%sigma0 = 0.1;
decay = 0.99;
%% parameters for DNN

W = 0*rand(p_N,dim,'gpuArray');

%% initial

%f = zeros(p_N,dim);

%loss_record = zeros(t_N,p_N);
B = 6*rand(1)-3;
%loss_record(1,:) = fcn_loss(W,B);
diff = ones(size(W),'gpuArray');
m = zeros(size(W),'gpuArray');
aver = zeros(size(W),'gpuArray');
%m_bias = zeros(size(W));
v = zeros(size(W),'gpuArray');
%v_bias = zeros(size(W));
loss = zeros(1,p_N);
%theta_rp = randperm(dim);
omega = zeros(p_N,1,'gpuArray');
%sum_omega  = zeros(p_N,1);
%theta_batch = dim;
tmp_index = repmat([1:1:p_N/p_batch],p_batch,1);
t = 0;
beta1 = 0.9;
beta2 = 0.999;
idx_i = [1:1:p_N];
idx_j = ceil(idx_i/p_batch);
M_trans = sparse(idx_i,idx_j,1);
Gradient = zeros(size(W),'gpuArray');
%while max(max(abs(diff)))>10^(-5)
%% main part
while t<5000
    t = t+1;
    p_rp = randperm(p_N);
    f_temp = fcn_loss(W,B);
    idx = reshape(p_rp,p_batch,[]);
    f_batch = f_temp(idx);
    f_min = min(f_batch);
    omega_temp = exp(-alpha*(f_batch-f_min));
    sum_omega = sum(omega_temp);
    omega = sparse(tmp_index,idx,double(omega_temp));

    aver_temp = omega*W./sum_omega';
    aver(p_rp,:) = M_trans*aver_temp;
    diff = W - aver;
    m = beta1*m+ (1-beta1)*diff ;
    v = beta2*v + (1-beta2)*diff.^2;
    m_bias = m/(1-beta1^t);
    v_bias = v/(1-beta2^t);
    Gradient = - lambda * m_bias./(sqrt(v_bias) + 1e-8)+  decay^(t/decay_rate)*(randn(size(m_bias),'gpuArray'));
    W = W +Gradient;
 
    if mod(t,500)==0
        %t
        %sprintf('diff: %f,m:%f',max(max(abs(diff))),max(max(abs(Gradient))))
        %sprintf('v:%f',max(max(abs(v_bias))))
        %sprintf('epoch =%d,error= %f',t ,max(abs(W(1,:)-B)))
    end
end
while max(max(abs(diff)))>10^(-1)
    t = t+1;
    p_rp = randperm(p_N);
    f_temp = fcn_loss(W,B);
    idx = reshape(p_rp,p_batch,[]);
    f_batch = f_temp(idx);
    f_min = min(f_batch);
    omega_temp = exp(-alpha*(f_batch-f_min));
    sum_omega = sum(omega_temp);
    omega = sparse(tmp_index,idx,double(omega_temp));

    aver_temp = omega*W./sum_omega';
    aver(p_rp,:) = M_trans*aver_temp;
    diff = W - aver;
    m = beta1*m+ (1-beta1)*diff ;
    v = beta2*v + (1-beta2)*diff.^2;
    m_bias = m/(1-beta1^t);
    v_bias = v/(1-beta2^t);
    Gradient = - lambda * m_bias./(sqrt(v_bias) + 1e-8)+  decay^(t/decay_rate)*(randn(size(m_bias),'gpuArray'));
    W = W +Gradient;
 
    if mod(t,500)==0
        %t
        %sprintf('diff: %f,m:%f',max(max(abs(diff))),max(max(abs(Gradient))))
        %sprintf('v:%f',max(max(abs(v_bias))))
        %sprintf('epoch =%d,error= %f',t ,max(abs(W(1,:)-B)))
    end
end
%sprintf('diff: %f,m:%f',max(max(abs(diff))),max(max(abs(Gradient))))

test = abs(gather(W(1,:))'-ones(dim,1)*B) < 0.25;

if sum(test) == dim
    vW_success_rate = vW_success_rate+1;
end
end
conv_success_rate = conv_success_rate/simu_N;
vW_success_rate = vW_success_rate/simu_N;
1;
