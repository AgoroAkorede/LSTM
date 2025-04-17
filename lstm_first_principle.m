clc;
load('Enhanced_Ph_Modelmat_2025_03_26.mat')

%----------Extract LSTM Parameters ------------
layer = ph_predictor.Layers(2);
input_size = layer.InputSize;
hidden_size = size(layer.Bias, 1)/4; % Each gate is 1/4th bias vector size

% ------------------- Load or Generate Test Data -----------------------
if exist('testData', 'var') % Check if test data exists in the .mat file
    X_test = testData; % Use test data from the .mat file
     disp('using real data from .mat file **************')
else
    % Generate random test data (replace with real data if available)
    num_time_steps = 7201; % Number of time steps
    X_test = randn(input_size, num_time_steps); % Random test data
    disp('using dummy data**************')
end

%----------Forget Gate ---------------
Wf = layer.InputWeights(1:hidden_size, :);           % Input Weight
Uf = layer.RecurrentWeights(1:hidden_size, :);       % Recurremt Weight
bf = layer.Bias(1:hidden_size);                      % Biases

%------------ Input Gate ------------
Wi = layer.InputWeights(hidden_size+1:2*hidden_size, :); % Rows H+1:2H
Ui = layer.RecurrentWeights(hidden_size+1:2*hidden_size, :);
bi = layer.Bias(hidden_size+1:2*hidden_size);

%------------- Output Gate ----------
Wo = layer.InputWeights(2*hidden_size+1:3*hidden_size, :); % Rows 2H+1:3H
Uo = layer.RecurrentWeights(2*hidden_size+1:3*hidden_size, :);
bo = layer.Bias(2*hidden_size+1:3*hidden_size);

%----------- Cell Candidate ---------
Wc = layer.InputWeights(3*hidden_size+1:4*hidden_size, :); % Rows 3H+1:4H
Uc = layer.RecurrentWeights(3*hidden_size+1:4*hidden_size, :);
bc = layer.Bias(3*hidden_size+1:4*hidden_size);

% ------------------- Initialize hidden and cell states -----------------------
h = zeros(hidden_size, 1);  % Initial hidden state
C = zeros(hidden_size, 1);  % Initial cell state

% ------------------- Forward Pass for Inference -----------------------
outputs = zeros(hidden_size, num_time_steps);

for t = 1:num_time_steps
    xt = X_test(:, t);

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

% ------------------- Save Outputs to a CSV File -----------------------
% Transpose outputs so that each row corresponds to a time step
outputs_csv = outputs'; % Transpose for row-wise storage

% Save to CSV file
filename = 'lstm_first_principle_outputs.csv'; % Name of the CSV file
writematrix(outputs_csv, filename); % Save outputs to CSV

disp(['LSTM outputs saved to: ', filename]);

function y = sigmoid(x)
    % Sigmoid activation function
    y = 1 ./ (1 + exp(-x));
end