function [pred_data_y, mse, mae] = run_lstm_prediction(neural_net, test_data_u, test_data_y, time)
% RUN_LSTM_PREDICTION - Run predictions on a trained LSTM network
%   This function extracts parameters from a trained LSTM network and
%   performs the forward pass dynamically for any network architecture
%
%   Inputs:
%   - neural_net: The trained LSTM network
%   - test_data_u: Test input data (format: features x time_steps)
%   - test_data_y: Test output data (format: features x time_steps)
%   - time: Time vector for plotting
%
%   Outputs:
%   - pred_data_y: Predicted values
%   - mse: Mean Squared Error
%   - mae: Mean Absolute Error

    % Get network parameters
    network_params = extract_network_parameters(neural_net);
    
    % Perform forward pass
    pred_data_y = lstm_forward_pass(network_params, test_data_u);
    
    % Calculate error metrics
    mse = mean((pred_data_y - test_data_y).^2);
    mae = mean(abs(pred_data_y - test_data_y));
    
    % Plot results
    plot_results(time(2:end), pred_data_y, test_data_y);
    
    % Display error metrics
    fprintf('Mean Squared Error: %.6f\n', mse);
    fprintf('Mean Absolute Error: %.6f\n', mae);
end
