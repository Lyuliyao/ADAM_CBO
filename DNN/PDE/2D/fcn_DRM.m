function loss = fcn_DRM(theta_W,theta_b) 
 x = 2*rand(1000,2,'single','gpuArray')-1;
 h = min(0.001*rand(size(x,1),2),abs(x));
 x_pp = fcn_DNN(x+h.*[1,0],theta_W,theta_b);
 x_p =  fcn_DNN(x,theta_W,theta_b);
 x_pm = fcn_DNN(x-h.*[1,0],theta_W,theta_b);
 D_x = (x_pp-x_pm)./(h(:,1)*2);
 y_pp = fcn_DNN(x+h.*[0,1],theta_W,theta_b);
 y_pm = fcn_DNN(x-h.*[0,1],theta_W,theta_b);
 D_y = (y_pp-y_pm)./(h(:,2)*2);
 loss = 4*sum(0.5*(x(:,1).^2).^(1/4).*D_x.*D_x)/1000;
 loss = loss + 4*sum(0.5*(x(:,2).^2).^(1/4).*D_y.*D_y)/1000;
 x = 2*rand(100,2)-1;
 x(:,1) = 0;
 loss = loss + 2 * sum(fcn_DNN(x,theta_W,theta_b))/100;
 x = 2*rand(100,2)-1;
 x(:,2) = 0;
 loss = loss + 2 * sum(fcn_DNN(x,theta_W,theta_b))/100;
 x = 2*rand(100,2)-1;
 x(1:50,1) = 1;
 x(50:end,1) = -1;
 loss = loss + sum((fcn_DNN(x,theta_W,theta_b)-sol_exact(x)).^2);
 x = 2*rand(100,2)-1;
 x(1:50,2) = 1;
 x(50:end,2) = -1;
 loss = loss + sum((fcn_DNN(x,theta_W,theta_b)-sol_exact(x)).^2);
end

        