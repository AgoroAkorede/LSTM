function dist = Dist_rejector(r_k,y_plant,W_dist)
%This simply compares the plant output with the predicted output and
%calculates the difference
e_k = y_plant - r_k;

error_min = -Inf; %Minimum permissible error
error_max = +Inf; %Maximum permissible error

if e_k > error_min && e_k < error_max 
    error_func = tanh(e_k);
    % error_func = (e_k);
else
    error_func = 0;
end


%The disturbance function is a linear function of the error function.

dist = (error_func*W_dist);


