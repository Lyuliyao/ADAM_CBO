function loss = fcn_error_2(theta_W,theta_b) 
 global dim;
 x = 2*rand(100,dim,'single','gpuArray')-1;
 y_true = sol_exact(x);
 y_pred = fcn_DNN(x,theta_W,theta_b);
 loss = sqrt(sum( (y_true-y_pred).^2 ))/sqrt(sum( (y_true).^2 ));
end
