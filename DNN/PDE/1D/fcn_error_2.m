function error_2 = fcn_error_2(theta_W,theta_b)
 x = 2*rand(100,1,'single','gpuArray')-1;
 y_true = sol_exact(x);
 y_pred = fcn_DNN(x,theta_W,theta_b);
 error_2 = sqrt(sum( (y_true-y_pred).^2 ))/sqrt(sum( (y_true).^2 ));
end

