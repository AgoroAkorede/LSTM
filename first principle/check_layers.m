function [all_LSTM_layers] = check_layers(layer, layer_number)
lstmNet = layer;
LSTM_layer_names = {};
for i = 1:layer_number
    % base_layer_name = 'lstm';
    LSTM_layer_name = sprintf('lstm_%d', i);
    LSTM_layer_names{end+1}= LSTM_layer_name;
end
% LSTM_layer_names ={'lstm_1', 'lstm_2', 'lstm_3'};
 all_LSTM_layers ={};
for i = 1:size(lstmNet.Layers)
    currentLayer = lstmNet.Layers(i);
    if ismember(currentLayer.Name, LSTM_layer_names)
       all_LSTM_layers{end+1} = currentLayer;
       % disp(currentLayer.Name)
    end
end

end