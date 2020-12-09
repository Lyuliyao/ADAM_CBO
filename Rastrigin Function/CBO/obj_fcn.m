function f = obj_fcn(W)

%%%%%%%%%%%%%%%%%%%%%% DNN %%%%%%%%%%%%%%%%%%%%%%
global X_train; global Y_train;
%f = fcn_DNN(X_train,Y_train,W);
%%%%%%%%%%%%%%%%%%%%%% DNN %%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%% Ackley function %%%%%%%%%%%%%%%%%%%%%%
% dim = size(W,1);
% B =0;  C = 0;
% diff = W-B;
% f = -20*exp(-0.2/sqrt(dim)*sqrt(sum((diff).^2,1))) ...
%     - exp(1/dim*sum(cos(2*pi*(W-B)),1)) ...
%     + 20 + exp(1) + C;
% %%%%%%%%%%%%%%%%%%%%%% Ackley function %%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%  function %%%%%%%%%%%%%%%%%%%%%%
 dim = size(W,1);
 global B; C = 0;
 diff = W-B;
 f = 1/dim*sum(diff.^2 - 10*cos(2*pi*diff) + 10,1) + C;
%%%%%%%%%%%%%%%%%%%%%% Ackley function %%%%%%%%%%%%%%%%%%%%%%




