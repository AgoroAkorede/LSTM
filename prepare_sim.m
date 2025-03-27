global pH_const

load('training_data');
load('training_data_simple');
load('enhanced_ph_model.mat')

pH_const = struct();
pH_const.ts = 5;
pH_const.net = pH_net;
pH_const.net_simple = pH_net_simple;
[x_0, u_0, d_0, y_0] = get_init_op();

% Try to sync network internal state. May not produce the desired result if network model is inaccurate  
nn_predict(y_0,y_0,u_0,u_0,1,50,0,true);
nn_simple_predict(u_0,1,50,0,true);

pH_const.u_0 = u_0; pH_const.x_0 = x_0; pH_const.d_0 = d_0; pH_const.y_0 = y_0;
pH_sim_data = sim("pH_control_loops");

% Uncomment or paste in command window to save simulation data
% save("simulation_data", "pH_sim_data");