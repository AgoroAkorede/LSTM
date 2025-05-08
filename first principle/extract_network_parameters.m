function params = extract_network_parameters(network, layers_info, num_hidden_units)
    params = struct();
    
    % Extract LSTM layer parameters
    for layer_idx = 1:layers_info.num_lstm_layers
        network_layer_idx = layers_info.lstm_indices(layer_idx);
        layer_name = sprintf('lstm%d', layer_idx);
        
        % Create structure for this layer
        params.(layer_name) = struct();
        
        % Extract weights and biases
        % Input gate
        params.(layer_name).Wi = network.Layers(network_layer_idx).InputWeights(1:num_hidden_units,:);
        params.(layer_name).Ui = network.Layers(network_layer_idx).RecurrentWeights(1:num_hidden_units,:);
        params.(layer_name).bi = network.Layers(network_layer_idx).Bias(1:num_hidden_units,:);
        
        % Forget gate
        params.(layer_name).Wf = network.Layers(network_layer_idx).InputWeights(num_hidden_units+1:2*num_hidden_units,:);
        params.(layer_name).Uf = network.Layers(network_layer_idx).RecurrentWeights(num_hidden_units+1:2*num_hidden_units,:);
        params.(layer_name).bf = network.Layers(network_layer_idx).Bias(num_hidden_units+1:2*num_hidden_units,:);
        
        % Cell gate
        params.(layer_name).Wc = network.Layers(network_layer_idx).InputWeights(2*num_hidden_units+1:3*num_hidden_units,:);
        params.(layer_name).Uc = network.Layers(network_layer_idx).RecurrentWeights(2*num_hidden_units+1:3*num_hidden_units,:);
        params.(layer_name).bc = network.Layers(network_layer_idx).Bias(2*num_hidden_units+1:3*num_hidden_units,:);
        
        % Output gate
        params.(layer_name).Wo = network.Layers(network_layer_idx).InputWeights(3*num_hidden_units+1:4*num_hidden_units,:);
        params.(layer_name).Uo = network.Layers(network_layer_idx).RecurrentWeights(3*num_hidden_units+1:4*num_hidden_units,:);
        params.(layer_name).bo = network.Layers(network_layer_idx).Bias(3*num_hidden_units+1:4*num_hidden_units,:);
    end
    
    % Extract fully connected layer parameters
    fc_idx = layers_info.fc_index;
    params.fc = struct();
    params.fc.Wy = network.Layers(fc_idx).Weights;
    params.fc.by = network.Layers(fc_idx).Bias;
    
    fprintf('Successfully extracted parameters for %d LSTM layers and the output layer\n', ...
        layers_info.num_lstm_layers);
end
