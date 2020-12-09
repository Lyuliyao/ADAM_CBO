 x = 2*rand(10,1)-1;
 y_p =  sol_exact(x);
 p = fittype('a*abs(x)^1.5+b*x+c*abs(x)^0.5+d');
 u = fit(x,y_p,p);
 co = coeffvalues(u);
 loss = 1.125*co(1)^2 +co(2)^2 +1.5*co(3)+ co(1)*(0.75 + 1.5*co(3))+3*co(4)
 %D_y = @(x) 3*co(1)*x./(2*(x.^2).^0.25)+ co(2) + co(3)*x./(2*(x.^2).^0.75);
 %f = @(x) -(3./(4*(x.^2).^(1/4)));
 %I =@(x) 0.5*D_y(x).^2 - f(x).*u(x);
 %integral(I,-1,1,'ArrayValued',true)
 %loss = 2*sum(0.5*D_y.^2  +3*y_p./(4*(x.^2).^0.25))/500