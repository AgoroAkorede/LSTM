clc;
clear;

% Load datasets
load('training_data');
base_model = "enhanced_ph_model.mat";
model_filename = format_model_name(base_model, true);

% Check for existing model
if exist(model_filename, 'file') == 1
    fprintf('Loading pre-trained model from: %s\n', model_filename);
    load(model_filename);
else
    fprintf('No existing model found. Starting training...\n');

    % Prepare training sequences
    [~, initial_flow] = get_init_op();

    flow_data = train_dataset.u.signals(1).values(:,1);
    ph_measurements = train_dataset.x.signals(3).values(:,1);

    num_samples = numel(flow_data);

    % Data preprocessing
    [X_train, train_settings] = mapminmax(flow_data(1:num_samples-1)');
    Y_train = ph_measurements(2:end)';
disp(X_train)
disp(flow_data(1:num_samples-1)')
    % Network configuration
    input_features = 1;
    output_features = 1;
    numHiddenUnits = 100;

    % Enhanced network architecture
    network_layers = [...
        sequenceInputLayer(input_features)
        lstmLayer(numHiddenUnits, 'OutputMode','sequence', 'StateActivationFunction','tanh', 'GateActivationFunction','sigmoid') 
        dropoutLayer(0.3) % Prevents Overfitting
        lstmLayer(numHiddenUnits)
         lstmLayer(numHiddenUnits)
        fullyConnectedLayer(50)
        reluLayer()
        fullyConnectedLayer(output_features)
        regressionLayer
    ];

    % Optimized training parameters
    training_config = trainingOptions('adam', ...
        'MaxEpochs', 5000, ...
        'MiniBatchSize', 64, ...
        'GradientThreshold', 1.5, ... % To prevent disappearing gradient
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 500, ...
        'LearnRateDropFactor', 0.7, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose', true);

    % Train and save model
    ph_predictor = trainNetwork(X_train, Y_train, network_layers, training_config);
    save(model_filename, 'ph_predictor', 'train_settings');
    fprintf('New model trained and saved as: %s\n', model_filename);
end

% Prepare test data
test_flow = test_dataset.u.signals(1).values(:,1);
test_ph = test_dataset.x.signals(3).values(:,1);
X_test = mapminmax('apply', test_flow(1:end-1)', train_settings);
Y_test = test_ph(2:end)';

% Generate predictions
ph_predictions = predict(ph_predictor, X_test);

% Performance metrics
mse = mean((ph_predictions - Y_test).^2);
fprintf('\nPerformance Metrics:\nMSE: %.4f\n', mse);

% Visualization
time_points = test_dataset.u.time(2:end)';
line_weight = 1.5;

figure('Color', 'white');
plot(time_points, ph_predictions, 'k-', 'LineWidth', line_weight)
hold on
plot(time_points, Y_test, 'r--', 'LineWidth', line_weight*0.9)
title('pH Prediction Performance', 'FontSize', 14, 'FontWeight', 'bold')
xlabel('Time (seconds)', 'FontSize', 12)
ylabel('pH Value', 'FontSize', 12)
legend({'Model Prediction', 'Experimental Data'}, 'Location', 'best')
grid on
axis tight