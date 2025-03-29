clc;
clear;

% Load datasets
load('training_data');
inverse_model_filename = "inverse_ph_model.mat";

% Check for existing inverse model
if exist(inverse_model_filename, 'file') == 1
    fprintf('Loading pre-trained inverse model from: %s\n', inverse_model_filename);
    load(inverse_model_filename);
else
    fprintf('No existing inverse model found. Starting inverse training...\n');

    % Prepare training sequences (Swapping input and output)
    [~, initial_flow] = get_init_op();

    flow_data = train_dataset.u.signals(1).values(:,1); % Previously input, now target
    ph_measurements = train_dataset.x.signals(3).values(:,1); % Previously output, now input

    num_samples = numel(flow_data);

    % Data preprocessing (Reversing the mapping direction)
    [X_train, inverse_train_settings] = mapminmax(ph_measurements(1:num_samples-1)');
    Y_train = flow_data(2:end)';

    % Network configuration
    input_features = 1;
    output_features = 1;

    num_of_hidden_units = 100;

    % Inverse response network architecture
    inverse_network_layers = [...  
        sequenceInputLayer(input_features) % SISO system requires Single Input Single Output
        lstmLayer(100, 'OutputMode', 'sequence', 'StateActivationFunction', 'tanh', 'GateActivationFunction', 'sigmoid') 
        dropoutLayer(0.3) % Prevents Overfitting
        lstmLayer(100)
        lstmLayer(100)
        fullyConnectedLayer(50)
        reluLayer()
        fullyConnectedLayer(output_features)
        regressionLayer
    ];

    % Optimized training parameters
    inverse_training_config = trainingOptions('adam', ...
        'MaxEpochs', 5000, ...
        'MiniBatchSize', 64, ...
        'GradientThreshold', 1.5, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 500, ...
        'LearnRateDropFactor', 0.7, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose', true);

    % Train and save inverse model
    inverse_ph_predictor = trainNetwork(X_train, Y_train, inverse_network_layers, inverse_training_config);
    save(inverse_model_filename, 'inverse_ph_predictor', 'inverse_train_settings');
    fprintf('New inverse model trained and saved as: %s\n', inverse_model_filename);
end

% Prepare test data (Reversing test input-output)
test_ph = test_dataset.x.signals(3).values(:,1); % pH as input
test_flow = test_dataset.u.signals(1).values(:,1); % Flow as target
X_test_inverse = mapminmax('apply', test_ph(1:end-1)', inverse_train_settings);
Y_test_inverse = test_flow(2:end)';

% Generate inverse predictions
flow_predictions = predict(inverse_ph_predictor, X_test_inverse);

% Performance metrics
mse_inverse = mean((flow_predictions - Y_test_inverse).^2);
fprintf('\nInverse Model Performance:\nMSE: %.4f\n', mse_inverse);

% Visualization
time_points = test_dataset.u.time(2:end)';
line_weight = 1.5;

figure('Color', 'white');
plot(time_points, flow_predictions, 'b-', 'LineWidth', line_weight)
hold on
plot(time_points, Y_test_inverse, 'g--', 'LineWidth', line_weight*0.9)
title('Inverse Model: Flow Prediction from pH', 'FontSize', 14, 'FontWeight', 'bold')
xlabel('Time (seconds)', 'FontSize', 12)
ylabel('Flow Rate', 'FontSize', 12)
legend({'Inverse Model Prediction', 'Actual Flow Data'}, 'Location', 'best')
grid on
axis tight
