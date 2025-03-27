% IMPORTANT
%
% A failed experiment, use train_lstm_simple instead
%
% If you want to replace the saved network with the network obtained from
% this training session, please execute the following in the command window
% after this script has finished executing
%
% save("training_data", "pH_net", "train_dataset", "test_dataset");

load('training_data');

% Comment this if the data set you want to use for training is already in the
% workspace
%
% train_dataset = sim("lstm_train_data");

[~, q3_init] = get_init_op();

q3 = train_dataset.u.signals(1).values(:,1);
pH = train_dataset.x.signals(3).values(:,1);

data_size = numel(q3);

q3_past_2 = [q3_init; q3_init; q3(1:data_size-2)];
q3_past = [q3_init; q3(1:data_size-1)];
pH_past_2 = [pH(1); pH(1); pH(1:data_size-2)];
pH_past = [pH(1); pH(1:data_size-1)];

train_data_u = [q3_past, q3_past_2, pH_past, pH_past_2]';
train_data_y = pH(1:data_size)';

train_data_u_dim = size(train_data_u);
train_data_y_dim = size(train_data_y);

net_inputs = train_data_u_dim(1);
net_outputs = train_data_y_dim(1);
num_hidden_units = 5;

layers = [...
    sequenceInputLayer(net_inputs)
    fullyConnectedLayer(num_hidden_units)
    fullyConnectedLayer(num_hidden_units)
    fullyConnectedLayer(net_outputs)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',0);

pH_net = trainNetwork(train_data_u, train_data_y, layers, options);

% Comment this if the data set you want to use for training is already in the
% workspace
%
% test_dataset = sim("lstm_test_data");

q3_test = test_dataset.u.signals(1).values(:,1);
pH_test = test_dataset.x.signals(3).values(:,1);

test_data_size = numel(q3_test);

q3_test_past_2 = [q3_init; q3_init; q3_test(1:test_data_size-2)];
q3_test_past = [q3_init; q3_test(1:test_data_size-1)];
pH_test_past_2 = [pH_test(1); pH_test(1); pH_test(1:test_data_size-2)];
pH_test_past = [pH_test(1); pH_test(1:test_data_size-1)];

test_data_u = [q3_test_past, q3_test_past_2, pH_test_past, pH_test_past_2]';
test_data_y = pH_test(1:test_data_size)';

time = test_dataset.u.time';

% Open loop
pred_data_y = predict(pH_net,test_data_u, 'MiniBatchSize', test_data_size);

LW = 1.4;

f1 = figure(1);
    plot(time,pred_data_y,'k-','LineWidth',LW)
    hold on
    plot(time,test_data_y,'r--','LineWidth',LW)
    xlabel('Time (s)')
    ylabel('pH')
    title('Validation Data (Open Loop)')
    grid on
    legend('Predicted', 'Actual')

% Closed loop
pred_data_y_closed = zeros(1, test_data_size);
pred_data_y_closed(1:2) = test_data_y(1:2);
pred_lstm = pH_net;
for i = 3:test_data_size
    net_input = [test_data_u(1,i-1);test_data_u(2,i-1);pred_data_y_closed(i-1);pred_data_y_closed(i-2)];
    [pred_lstm, y_pred] = predictAndUpdateState(pred_lstm,net_input);
    pred_data_y_closed(i) = y_pred;
end
f2 = figure(2);
    plot(time,pred_data_y_closed,'k-','LineWidth',LW)
    hold on
    plot(time,test_data_y,'r--','LineWidth',LW)
    xlabel('Time (s)')
    ylabel('pH')
    title('Validation Data (Closed Loop)')
    grid on
    legend('Predicted', 'Actual')
