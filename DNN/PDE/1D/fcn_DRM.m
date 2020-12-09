function loss = fcn_DRM(theta_W,theta_b)
%FCN_DRM Summary of this function goes here
%   Detailed explanation goes here
 x = 2*rand(1000,1,'single','gpuArray')-1;
 h = min(0.01*rand(size(x,1),1),abs(x));
 y_pp = fcn_DNN(x+h,theta_W,theta_b);
 y_p =  fcn_DNN(x,theta_W,theta_b);
 y_pm = fcn_DNN(x-h,theta_W,theta_b);
 D_y = (y_pp-y_pm)./(h*2);
 loss = 2*sum(0.5*(x.^2).^(1/4).*D_y.^2)/1000+fcn_DNN(0,theta_W,theta_b);
end

