function x = Figuredata()
    load('approximatefunction.mat');
    x = -1:0.01:1;
    [Value,Index] = min(error_record_inf(t,:));
    theta_W = theta_W_record(Index,:);
    theta_b = theta_b_record(Index,:);
    y_true=sol_exact(x');
    y_pred=DNN(x',theta_W,theta_b)';
    save('Figuredata.mat','x','y_true','y_pred');
end