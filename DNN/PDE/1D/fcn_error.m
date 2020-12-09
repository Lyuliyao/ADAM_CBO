function loss = fcn_error(theta_W,theta_b) 
 x = 2*rand(100,1,'gpuArray')-1;
 y_true = sol_exact(x);
 y_pred = DNN(x,theta_W,theta_b);
 loss = sqrt(sum( (y_true-y_pred).^2 ))/sqrt(sum( (y_true).^2 ));
end

        