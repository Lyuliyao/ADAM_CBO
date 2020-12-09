N_W = 0;
N_b = 0;
for i = 1:size(Layer,2)-1
    N_W = N_W + Layer(i)*Layer(i+1);
    N_b = N_b + Layer(i+1);
end
theta_W_record = 2*rand(p_N,N_W)-1;
theta_b_record = 2*rand(p_N,N_b)-1;
