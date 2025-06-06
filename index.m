clc;


% % % % % % % % % % % % % % % 
% FOR PH SISO
load('training_data_simple');
name ="gru_ph_model";

[~, q3_init] = get_init_op();
% INPUT
q3 = train_dataset.u.signals(1).values(:,1);
train_data_u = [q3_init,q3(1:data_size-1)'];
train_data_u_dim = size(train_data_u);
net_inputs = train_data_u_dim(1);
% OUTPUT
pH = train_dataset.x.signals(3).values(:,1);
train_data_y = pH(1:end)';
train_data_y_dim = size(train_data_y);
net_outputs = train_data_y_dim(1);
% 
data_size = numel(q3);
num_hidden_units = 50;

% Define te GRU newtork Layer

layers = [
    sequenceInputLayer(net_inputs, 'Name', 'InputLayer')
    gruLayer(num_hidden_units, 'OutputMode', 'sequence', 'Name', 'GRU1') % Increase units
    dropoutLayer(0.1, 'Name', 'Dropout1') 
    gruLayer(num_hidden_units, 'OutputMode', 'sequence', 'Name', 'GRU2') % Additional GRU layer
    dropoutLayer(0.1, 'Name', 'Dropout2') 
     gruLayer(num_hidden_units, 'OutputMode', 'sequence', 'Name', 'GRU3') % Additional GRU layer
    dropoutLayer(0.1, 'Name', 'Dropout3') 
     gruLayer(num_hidden_units, 'OutputMode', 'sequence', 'Name', 'GRU4') % Additional GRU layer
    dropoutLayer(0.1, 'Name', 'Dropout4') 
    fullyConnectedLayer(net_outputs, 'Name', 'FullyConnected') 
    regressionLayer('Name', 'Output')];


% Test data
q3_test = test_dataset.u.signals(1).values(:,1);
pH_test = test_dataset.x.signals(3).values(:,1);

test_data_size = numel(q3_test);

test_data_u = [q3_init,q3_test(1:test_data_size-1)'];
test_data_y = pH_test(1:end)';

time = test_dataset.u.time';


options = trainingOptions('adam', ...
    'MaxEpochs', 5e3, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ... % Ensure the learning rate is not too high
    'Shuffle', 'every-epoch', ...
    'ValidationData', {test_data_u, test_data_y}, ...
    'ValidationFrequency', 10, ...
    'GradientThreshold', 1, ... % Prevent exploding gradients
      'ExecutionEnvironment', 'cpu', ... % Enforce CPU usage
    'Verbose', false, ...
    'Plots', 'training-progress');

% % Train the GRU Model
GRUPHNet = trainNetwork(train_data_u, train_data_y, layers, options);
save(name,'GRUPHNet')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE MSE for training process/data

pred_train_data_y = predict(GRUPHNet,train_data_u);
mseValue=mean(pred_train_data_y-train_data_y).^2;
disp(['MSE:', num2str(mseValue)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


LW = 1.4;
f1 = figure(1);
    plot(time(1:end),pred_train_data_y,'k-','LineWidth',LW)
    hold on
    plot(time(1:end),test_data_y,'r--','LineWidth',LW)
    xlabel('Time (s)')
    ylabel('pH')
    grid on
    legend('Predicted', 'Actual')
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Closed-Loop Simulation and Performance Evaluation

% Initialize predicted pH values for closed-loop simulation
pred_data_y_closed = zeros(1, test_data_size); % Matrix to store predicted pH values
pred_data_y_closed(1:2) = test_data_y(1:2); % Use true pH values for the first two time steps

% Clone the trained network for stateful prediction
pred_GRU = GRUPHNet;

% Perform closed-loop prediction
for i = 3:test_data_size
    % Construct input for the model at the current time step
    % Include external inputs (e.g., flow rates) and previous predicted pH values
    net_input = [test_data_u(i-1); pred_data_y_closed(i-1); pred_data_y_closed(i-2)];
    
    % Predict the next pH value and update the network state
    [pred_GRU, y_pred] = predictAndUpdateState(pred_GRU, net_input);
    
    % Store the predicted pH value for the current time step
    pred_data_y_closed(i) = y_pred;
end

% Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for closed-loop predictions
mse_closed = mean((pred_data_y_closed - test_data_y).^2, 'all'); % MSE for closed-loop simulation
rmse_closed = sqrt(mse_closed); % RMSE for closed-loop simulation

disp(['Closed-Loop MSE: ', num2str(mse_closed)]);
disp(['Closed-Loop RMSE: ', num2str(rmse_closed)]);

% Visualize closed-loop predictions vs. actual pH values
f2 = figure(2);
plot(time, pred_data_y_closed, 'k-', 'LineWidth', LW); hold on; % Predicted pH values
plot(time, test_data_y, 'r--', 'LineWidth', LW); % Actual pH values
xlabel('Time (s)'); % Label for x-axis
ylabel('pH'); % Label for y-axis
title('Validation Data (Closed Loop)'); % Title of the plot
grid on; % Enable grid for better readability
legend('Predicted', 'Actual'); % Legend for the plot