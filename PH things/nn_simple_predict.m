function y_k = nn_simple_predict(u_k,Nu,Np,e,update_state)
    global pH_const;
    session_net = pH_const.net_simple;
    y_k = zeros(1, Np);
    for i = 1:Np
        if (i == 1)
            u = u_k(1);
        elseif (i <= Nu)
            u = u_k(i);
        else
            u = u_k(Nu);
        end
        session_u = prepare_input_data(u,pH_const.norm_settings,pH_const.seq_length);
        % fprintf('This is the flow rate : %s === This is the prepared flow rate : %s \n',u,session_u)
        [session_net, y_ki] = predictAndUpdateState(session_net,session_u);
%         [session_net, y_ki] = predictAndUpdateState(session_net,u);
        y_k(:,i) = y_ki  + e;
    end
    if (update_state)
       pH_const.net_simple = session_net;
    end
end