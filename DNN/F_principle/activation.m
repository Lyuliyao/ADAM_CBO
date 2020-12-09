function x =activation(x)   
x = 1./(1+exp(-x));
%x = max(x,0);
end