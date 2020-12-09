function vW_success_rate = swarming_general_onepara(dim,p_N,p_batch,random,lambda,gama,sigma,simu_N)
alpha = 30;
tN = 10000; t_now = 0;
simu_now  = 0;
eps = 0.01;
conv_success_rate = 0;
vW_success_rate = 0;
global B; B = 6*rand(1)-3;
%%%%%%%%%%%%%%% partial batch of obj fcn %%%%%%%%%%%%%%

%initialization of the weights
global para_N; para_N = 1; %dim = 20;
fv = zeros(simu_N);
v_W = ones(dim);
er = zeros(simu_N);

tic
for simu = simu_now+1:simu_N
%     sigma = sigma + 0.001;
jump_number = 0; fv_jump = 0; % this is for stop criteria when there is jump
W =  rand(dim,p_N)*6-3; 
f = zeros(1,p_N);
% while epoch < 101 %this is for partial batch
t = 1;
p_rp = randperm(p_N);
p_before = [];
f_temp = gather(obj_fcn(W));
for t = 1:tN %this is for full batch
    W_old = W; % this is for Criteria 2
        %calculate f_j, for j \in B
        %permute and divide
        if length(p_rp) < p_batch
            f_temp = gather(obj_fcn(W));
            p_rp = [p_rp, randperm(p_N)];
            p_before = [];
        end
%         idx = p_rp(1:p_batch);
        idx = p_rp(1:p_batch);
        p_rp(1:p_batch) = [];
        
        
        %%%%%%%%%%%%%%% full batch of obj fcn %%%%%%%%%%%%%%%
        f= f_temp(idx);
        %%%%%%%%%%%%%%% full batch of obj fcn %%%%%%%%%%%%%%%
            
        %%%%%%%%%%%%%%% partial batch of obj fcn %%%%%%%%%%%%%%%
%         if k > floor(X_N/X_batch)
%             k = 1; epoch = epoch +1;
%         end
%         f(idx) = obj_fcn_batch(W(:,idx),k);
%         k = k+1;
        %%%%%%%%%%%%%%% partial batch of obj fcn %%%%%%%%%%%%%%%

        %calculate v_W and fv
        f_min = min(f);
        omega = exp(-alpha*(f_temp-f_min));
        sum_omega = sum(omega);
        v_W = sum(W .* omega,2)/sum_omega;
        diff = W - v_W;
        W = W-lambda*gama*diff;
        if random == "normal"
            W = W + sqrt(2*gama)*sigma*diff.*randn(size(diff)); %this is for the origin method 
        elseif random == "uniform"
            W = W + sqrt(2*gama)*sigma*diff.*(2*rand(size(diff))-1);
        elseif random == "levy"
            if t ==1
                R = sqrt(1/tN) *randn(size(diff));
            else
                R = R +  sqrt(1/tN)*randn(size(diff));
            end
            W = W + sqrt(2*gama)*sigma*diff.*R;
        end
end

test = abs(v_W-ones(dim,1)*B) < 0.25;
if sum(test) == dim
    vW_success_rate = vW_success_rate+1;
end




end

toc

conv_success_rate = conv_success_rate/simu_N;
vW_success_rate = vW_success_rate/simu_N;
1;





    


