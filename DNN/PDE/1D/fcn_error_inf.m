function error_inf = fcn_error_inf(theta_W,theta_b)
 x = 2*rand(100,1,'single','gpuArray')-1;
 x(1,:) = 0;
 y_true = sol_exact(x);
 y_pred = fcn_DNN(x,theta_W,theta_b);
 error_inf = max(abs(y_true-y_pred));
end


