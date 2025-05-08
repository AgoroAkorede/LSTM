function layers_info = extract_lstm_layers(network)
    % Find all LSTM layers in the network
    lstm_layer_indices = [];
    for i = 1:length(network.Layers)
        if contains(class(network.Layers(i)), 'lstm', 'IgnoreCase', true)
            lstm_layer_indices = [lstm_layer_indices, i];
        end
    end
    
    % Get the fully connected layer index (assumes it follows the last LSTM layer)
    fc_layer_idx = lstm_layer_indices(end) + 1;
    
    % Create structure to hold layer information
    layers_info = struct();
    layers_info.num_lstm_layers = length(lstm_layer_indices);
    layers_info.lstm_indices = lstm_layer_indices;
    layers_info.fc_index = fc_layer_idx;
    
    fprintf('Found %d LSTM layers at indices: %s\n', ...
        layers_info.num_lstm_layers, ...
        num2str(lstm_layer_indices));
    fprintf('Fully connected layer at index: %d\n', fc_layer_idx);
end

