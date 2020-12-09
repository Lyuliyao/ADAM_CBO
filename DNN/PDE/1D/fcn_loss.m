function loss = fcn_loss(theta_W,theta_b) 
 x = 2*rand(1000,1)-1;
 h = min(0.01*rand(size(x,1),1),abs(x));
 y_pp = DNN(x+h,theta_W,theta_b);
 y_p =  DNN(x,theta_W,theta_b);
 y_pm = DNN(x-h,theta_W,theta_b);
 D_y = (y_pp-y_pm)./(h*2);
 loss = 2*sum(0.5*(x.^2).^(1/4).*D_y.^2)/1000+DNN(0,theta_W,theta_b);
 %loss = sum( (Laplace_p - 2).^2 )/201;
 %loss = loss + 500*((DNN(-1,theta_W,theta_b)-1)^2 + (DNN(1,theta_W,theta_b)-1)^2);
end

        