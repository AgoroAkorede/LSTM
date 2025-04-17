clc;
clear;

% ------------------- Load Pre-Trained Model -----------------------
load('Enhanced_Ph_Modelmat_2025_03_26.mat'); % Load pre-trained LSTM model
load('training_data');

% ------------------- Extract LSTM Parameters -----------------------
layer = ph_predictor.Layers(2); % Assuming LSTM is the 2nd layer
input_size = layer.InputSize;    % Number of input features
hidden_size = size(layer.Bias, 1) / 4; % H = hidden_size

% Forget Gate
Wf = layer.InputWeights(1:hidden_size, :);           % Rows 1:H
Uf = layer.RecurrentWeights(1:hidden_size, :);       % Rows 1:H
bf = layer.Bias(1:hidden_size);                      % First H elements

% Input Gate
Wi = layer.InputWeights(hidden_size+1:2*hidden_size, :); % Rows H+1:2H
Ui = layer.RecurrentWeights(hidden_size+1:2*hidden_size, :);
bi = layer.Bias(hidden_size+1:2*hidden_size);

% Output Gate
Wo = layer.InputWeights(2*hidden_size+1:3*hidden_size, :); % Rows 2H+1:3H
Uo = layer.RecurrentWeights(2*hidden_size+1:3*hidden_size, :);
bo = layer.Bias(2*hidden_size+1:3*hidden_size);

% Cell Candidate
Wc = layer.InputWeights(3*hidden_size+1:4*hidden_size, :); % Rows 3H+1:4H
Uc = layer.RecurrentWeights(3*hidden_size+1:4*hidden_size, :);
bc = layer.Bias(3*hidden_size+1:4*hidden_size);

% ------------------- Define Sigmoid and Tanh Functions -----------------------
function y = sigmoid(x)
    % Sigmoid activation function
    y = 1 ./ (1 + exp(-x));
end

function y = tanh(x)
    % Hyperbolic tangent activation function
    y = 2 ./ (1 + exp(-2 * x)) - 1;
end

% ------------------- Extract Test Data -----------------------
% Ensure test_dataset is loaded
if ~exist('test_dataset', 'var')
    error('Variable "test_dataset" is not defined. Please load your dataset.');
end

q3_test = test_dataset.u.signals(1).values(:, 1); % Input signal (e.g., flow rate)
pH_test = test_dataset.x.signals(3).values(:, 1); % Output signal (e.g., pH)

% Define test data size
test_data_size = numel(q3_test);

% Prepare test data for LSTM
test_data_u = q3_test(1:test_data_size-1)'; % Input sequence (excluding last value)
test_data_y = pH_test(2:end)';             % Target sequence (excluding first value)

% Normalize test data (if normalization was applied during training)
X_test = mapminmax('apply', test_data_u, train_settings);

% Extract time vector
time = test_dataset.u.time'; % Time vector

% ------------------- Initialize States -----------------------
h = zeros(hidden_size, 1);            % Initial hidden state
C = zeros(hidden_size, 1);            % Initial cell state
outputs = zeros(hidden_size, test_data_size - 1); % Preallocate outputs

% ------------------- Perform Inference -----------------------
for t = 1:(test_data_size - 1)
    xt = X_test(:, t); % Current input
    
    % Forget gate
    ft = sigmoid(Wf * xt + Uf * h + bf);
    
    % Input gate
    it = sigmoid(Wi * xt + Ui * h + bi);
    
    % Cell candidate
    Ct_candidate = tanh(Wc * xt + Uc * h + bc);
    
    % Update cell state
    C = ft .* C + it .* Ct_candidate;
    
    % Output gate
    ot = sigmoid(Wo * xt + Uo * h + bo);
    
    % Hidden state
    h = ot .* tanh(C);
    
    % Store output
    outputs(:, t) = h;
end

% Extract predicted values (use the first hidden unit as prediction)
pred_data_y = outputs(1, :); % Predicted pH values

% ------------------- Save Outputs to CSV -----------------------
filename = 'lstm_first_principle_outputs.csv'; % Name of the CSV file

if exist(filename, 'file') == 2
    % File exists: Append a timestamp to create a new file
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS'); % Get current timestamp
    new_filename = strcat('lstm_outputs_', timestamp, '.csv'); % New filename
    writematrix(outputs', new_filename); % Save outputs to a new file
    disp(['New file "', new_filename, '" has been created.']);
else
    % File does not exist: Save the outputs directly
    writematrix(outputs', filename); % Save outputs to CSV
    disp(['File "', filename, '" has been created.']);
end

% ------------------- Plot Predicted vs Actual -----------------------
LW = 1.4; % Line width
fig1 = figure(1);

% Ensure all vectors are row vectors
time = time(:)';         % Convert to row vector if necessary
pred_data_y = pred_data_y(:)'; % Convert to row vector if necessary
test_data_y = test_data_y(:)'; % Convert to row vector if necessary

% Debugging: Check sizes of time and pred_data_y
disp(['Length of time: ', num2str(length(time))]);
disp(['Length of pred_data_y: ', num2str(length(pred_data_y))]);

% Ensure lengths match
min_length = min(length(time), length(pred_data_y));
time = time(1:min_length);
pred_data_y = pred_data_y(1:min_length);

test_data_y = test_data_y(1:min_length);
 %---------- Input Range -------------
disp(['Min/max of X_test: ', num2str(min(X_test)), ' to ', num2str(max(X_test))]);

disp(['Min/max of pred_data_y: ', num2str(min(pred_data_y)), ' to ', num2str(max(pred_data_y))]);
disp(['Min/max of test_data_y: ', num2str(min(test_data_y)), ' to ', num2str(max(test_data_y))]);

% Plot predicted values
plot(time(2:end), pred_data_y(2:end), 'k-', 'LineWidth', LW); % Predicted values
hold on;

% Plot actual values
plot(time(2:end), test_data_y(2:end), 'r--', 'LineWidth', LW); % Actual values

% Add labels and legend
xlabel('Time (s)');
ylabel('pH');
title('LSTM Predictions vs Actual Values');
grid on;
legend('Predicted', 'Actual', 'Location', 'best'); % Add legend
hold off;