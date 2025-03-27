function y_k = nn_predict(y_past,y_present,u_past,u_k,Nu,Np,e,update_state)
    global pH_const;
    session_net = pH_const.net;
    y_k = zeros(1, Np);
    for i = 1:Np
        if (i == 1)
            u = [u_k(1),u_past];
        elseif (i <= Nu)
            u = [u_k(i),u_k(i-1)];
        else
            u = [u_k(Nu),u_k(Nu)];
        end
        if (i == 1)
            y = [y_present,y_past];
        elseif (i == 2)
            y = [y_k(:,1),y_present];
        else
            y = [y_k(:,i-1),y_k(:,i-2)];
        end
%         Compensation for silly ordering mistake during training  
        y = [y(2),y(1)];
        u = [u(2),u(1)];
        [session_net, y_ki] = predictAndUpdateState(session_net,[u(1);u(2);y(1);y(2)]);
        y_k(:,i) = y_ki  + e;
    end
    if (update_state)
       pH_const.net = session_net; 
    end
end