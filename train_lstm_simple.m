% IMPORTANT
%
% If you want to replace the saved network with the network obtained from
% this training session, please execute the following in the command window
% after this script has finished executing
%
% save("training_data_simple", "pH_net_simple", "train_dataset", "test_dataset");

load('training_data_simple');

% Comment this if the data set you want to use for training is already in the
% workspace
%
% train_dataset = sim("lstm_train_data");

[~, q3_init] = get_init_op();

q3 = train_dataset.u.signals(1).values(:,1);
pH = train_dataset.x.signals(3).values(:,1);

data_size = numel(q3);

train_data_u = q3(1:data_size-1)';
train_data_y = pH(2:end)';

train_data_u_dim = size(train_data_u);
train_data_y_dim = size(train_data_y);

net_inputs = train_data_u_dim(1);
net_outputs = train_data_y_dim(1);
num_hidden_units = 10;

layers = [...
    sequenceInputLayer(net_inputs)
    fullyConnectedLayer(num_hidden_units)
    lstmLayer(5)
    fullyConnectedLayer(num_hidden_units)
    fullyConnectedLayer(net_outputs)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.016, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',0);

pH_net_simple = trainNetwork(train_data_u, train_data_y, layers, options);

% Comment this if the data set you want to use for training is already in the
% workspace
%
% test_dataset = sim("lstm_test_data");

q3_test = test_dataset.u.signals(1).values(:,1);
pH_test = test_dataset.x.signals(3).values(:,1);

test_data_size = numel(q3_test);

test_data_u = q3_test(1:test_data_size-1)';
test_data_y = pH_test(2:end)';

time = test_dataset.u.time';

pred_data_y = predict(pH_net_simple,test_data_u);

LW = 1.4;
f1 = figure(1);
    plot(time(2:end),pred_data_y,'k-','LineWidth',LW)
    hold on
    plot(time(2:end),test_data_y,'r--','LineWidth',LW)
    xlabel('Time (s)')
    ylabel('pH')
    grid on
    legend('Predicted', 'Actual')