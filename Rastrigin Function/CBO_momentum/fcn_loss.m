function f = fcn_loss(x,B)
    n = size(x, 2);
    f =  sum((x-B) .^2 - 10 * cos(2 * pi * (x-B))+10, 2)/n;
end