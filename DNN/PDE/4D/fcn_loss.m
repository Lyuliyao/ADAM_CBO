function loss = fcn_loss(theta_W,theta_b) 
 global dim;
 x = 2*rand(1000,dim)-1;
 h = min(0.001*rand(size(x,1),dim),abs(x)/2);
 I = eye(dim);
 loss = 0 ;
 for i = 1:dim
    x_pp = DNN(x+h.*I(i,:),theta_W,theta_b);
    x_p =  DNN(x,theta_W,theta_b);
    x_pm = DNN(x-h.*I(i,:),theta_W,theta_b);
    D_x = (x_pp-x_pm)./(h(:,i)*2);
    loss = loss +  2^dim *sum(0.5*(x(:,i).^2).^(1/4).*D_x.*D_x)/1000;
  end
 for i = 1:dim
     x = 2*rand(100,dim)-1;
     x(:,i) = 0;
     loss = loss + 2^(dim-1) * sum(DNN(x,theta_W,theta_b))/100;
 end
 for i = 1:dim
     x = 2*rand(100,dim)-1;
     x(1:50,i) = 1;
     x(50:end,i) = -1;
     loss = loss + 50*sum((DNN(x,theta_W,theta_b)-sol_exact(x)).^2);
 end
end

        