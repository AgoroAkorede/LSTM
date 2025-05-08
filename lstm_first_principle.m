clc;
clear;

% Load data and model
load('training_data_simple.mat');
load('Enhanced_Ph_Modelmat_2025_04_26.mat'); % Load trained model

% Prepare test data
test_flow = test_dataset.u.signals(1).values(:, 1);
test_ph = test_dataset.x.signals(3).values(:, 1);
X_test = mapminmax('apply', test_flow(1:end-1)', train_settings);
Y_test = test_ph(2:end)';

test_data_size = numel(test_flow);
test_data_u = test_flow(1:test_data_size-1)';
test_data_y = test_ph(2:end)';
time = test_dataset.u.time';


% Extract network parameters
net_inputs=1;
numHiddenUnits = 100;
lstm1 = ph_predictor.Layers(2); % First LSTM layer
lstm2 = ph_predictor.Layers(4); % Second LSTM layer
lstm3 = ph_predictor.Layers(5); % Third LSTM layer
fcLayer = ph_predictor.Layers(7); % Fully connected layer


% ==================== First LSTM Layer Weights ====================
W1 = lstm1.InputWeights;
U1 = lstm1.RecurrentWeights;
b1 = lstm1.Bias;

Wi1 = W1(1:numHiddenUnits, :);
Wf1 = W1(numHiddenUnits+1:2*numHiddenUnits, :);
Wc1 = W1(2*numHiddenUnits+1:3*numHiddenUnits, :);
Wo1 = W1(3*numHiddenUnits+1:end, :);

Ui1 = U1(1:numHiddenUnits, :);
Uf1 = U1(numHiddenUnits+1:2*numHiddenUnits, :);
Uc1 = U1(2*numHiddenUnits+1:3*numHiddenUnits, :);
Uo1 = U1(3*numHiddenUnits+1:end, :);

bi1 = b1(1:numHiddenUnits);
bf1 = b1(numHiddenUnits+1:2*numHiddenUnits);
bc1 = b1(2*numHiddenUnits+1:3*numHiddenUnits);
bo1 = b1(3*numHiddenUnits+1:end);

% ==================== Second LSTM Layer Weights ====================
W2 = lstm2.InputWeights;
U2 = lstm2.RecurrentWeights;
b2 = lstm2.Bias;

Wi2 = W2(1:numHiddenUnits, :);
Wf2 = W2(numHiddenUnits+1:2*numHiddenUnits, :);
Wc2 = W2(2*numHiddenUnits+1:3*numHiddenUnits, :);
Wo2 = W2(3*numHiddenUnits+1:end, :);

Ui2 = U2(1:numHiddenUnits, :);
Uf2 = U2(numHiddenUnits+1:2*numHiddenUnits, :);
Uc2 = U2(2*numHiddenUnits+1:3*numHiddenUnits, :);
Uo2 = U2(3*numHiddenUnits+1:end, :);

bi2 = b2(1:numHiddenUnits);
bf2 = b2(numHiddenUnits+1:2*numHiddenUnits);
bc2 = b2(2*numHiddenUnits+1:3*numHiddenUnits);
bo2 = b2(3*numHiddenUnits+1:end);

% ==================== Third LSTM Layer Weights ====================
W3 = lstm3.InputWeights;
U3 = lstm3.RecurrentWeights;
b3 = lstm3.Bias;

Wi3 = W3(1:numHiddenUnits, :);
Wf3 = W3(numHiddenUnits+1:2*numHiddenUnits, :);
Wc3 = W3(2*numHiddenUnits+1:3*numHiddenUnits, :);
Wo3 = W3(3*numHiddenUnits+1:end, :);

Ui3 = U3(1:numHiddenUnits, :);
Uf3 = U3(numHiddenUnits+1:2*numHiddenUnits, :);
Uc3 = U3(2*numHiddenUnits+1:3*numHiddenUnits, :);
Uo3 = U3(3*numHiddenUnits+1:end, :);

bi3 = b3(1:numHiddenUnits);
bf3 = b3(numHiddenUnits+1:2*numHiddenUnits);
bc3 = b3(2*numHiddenUnits+1:3*numHiddenUnits);
bo3 = b3(3*numHiddenUnits+1:end);

% Fully connected layer
Wy = fcLayer.Weights;
by = fcLayer.Bias;

% Initialize states for all layers
h1 = zeros(numHiddenUnits, 1);
C1 = zeros(numHiddenUnits, 1);
h2 = zeros(numHiddenUnits, 1);
C2 = zeros(numHiddenUnits, 1);
h3 = zeros(numHiddenUnits, 1);
C3 = zeros(numHiddenUnits, 1);

% % Manual prediction storage
% pH_pred = zeros(size(X_test, 2), 1);
testData = test_data_u;
[num_features, num_time_steps] = size(testData);

pH_pred = zeros(net_inputs, num_time_steps);
% Activation functions
sigmoid = @(x) 1./(1+exp(-x));

% testData=X_test;

% Initialize an array to store the outputs at each time step
% pH_pred = zeros(net_inputs, num_time_steps);
outputSequence = zeros(numHiddenUnits, num_time_steps);

% Generate predictions
ph_predictions = predict(ph_predictor, X_test);

% Time step loop
for t = 1:num_time_steps
    xt = testData(:, t);
    % ==================== First LSTM Layer ====================
    f1 = sigmoid(Wf1*xt + Uf1*h1 + bf1);
    i1 = sigmoid(Wi1*xt + Ui1*h1 + bi1);

      % Candidate cell state (layer 1)
    C1_candidate = tanh(Wc1 * xt + Uc1 * h1 + bc1);
    C1 = f1.*C1 + i1.*C1_candidate;
    o1 = sigmoid(Wo1 *xt + Uo1 * h1 + bo1);
    h1 = o1 .* tanh(C1);
    % ==================== Second LSTM Layer ====================
    f2 = sigmoid(Wf2 * h1 + Uf2 * h2 + bf2);
    i2 = sigmoid(Wi2 * h1 + Ui2 * h2 + bi2);
   % Candidate cell state (layer 2)
    C2_candidate = tanh(Wc2 * h1 + Uc2 * h2 + bc2);
    C2 = f2.*C2 + i2.*C2_candidate;
    o2 = sigmoid(Wo2 * h1 + Uo2 * h2 + bo2);
    h2 = o2 .* tanh(C2);
    
    % ==================== Third LSTM Layer =================
    f3 = sigmoid(Wf3*h2 + Uf3*h3 + bf3);
    i3 = sigmoid(Wi3*h2 + Ui3*h3 + bi3);
    C3_Candidate = tanh(Wc3 * h2 + Uc3 * h3 + bc3);
    C3 = f3.*C3 + i3.*C3_Candidate;
    o3 = sigmoid(Wo3*h2 + Uo3*h3 + bo3);
    h3 = o3 .* tanh(C3);
    % ==================== Output Layer ====================
    ypred = Wy * h3 + by;

    % ypred = Wy * h2 + by;
    
    % Denormalize prediction
    pH_pred(:, t)= ypred;

    % fprintf('Time step %d: ypred = %.4f\n', t, ypred);
end


% % Example of correct denormalization:
% pH_pred(t) = ypred * (train_settings.ymax - train_settings.ymin) + train_settings.ymin;
% Check training data min/max

% Plot results
line_width= 1.5;
figure;
hold on;
% plot(pH_pred, 'k-', 'LineWidth', line_width);
plot(test_data_y, 'r--', 'LineWidth',line_width * 0.9);
plot(ph_predictions, 'k-', 'LineWidth',line_width * 0.9);
legend( 'Trained Model', 'Actual Data','PH Predictions', 'Location','best');
xlabel('Time Step', 'FontSize',12);
ylabel('pH Value', 'FontSize',12);
title('pH Prediction Comparison', 'FontSize',14, 'FontWeight','bold');
grid on;
axis tight;