clc;
clear;

% Load model data
load('Enhanced_Ph_Modelmat_2055_05_06.mat');

% Define constants
NET_INPUTS = 1;
NUM_HIDDEN_UNITS = 10;
NUM_LSTM_LAYERS = 3;

% Get the LSTM network
lstmNet = pH_net_simple;

% Extract all LSTM layers from the network
all_LSTM_layers = check_layers(lstmNet, NUM_LSTM_LAYERS);

% Initialize cell arrays to store weights and biases for each layer
Wi_all = cell(1, NUM_LSTM_LAYERS);
Ui_all = cell(1, NUM_LSTM_LAYERS);
bi_all = cell(1, NUM_LSTM_LAYERS);
Wf_all = cell(1, NUM_LSTM_LAYERS);
Uf_all = cell(1, NUM_LSTM_LAYERS);
bf_all = cell(1, NUM_LSTM_LAYERS);
Wc_all = cell(1, NUM_LSTM_LAYERS);
Uc_all = cell(1, NUM_LSTM_LAYERS);
bc_all = cell(1, NUM_LSTM_LAYERS);
Wo_all = cell(1, NUM_LSTM_LAYERS);
Uo_all = cell(1, NUM_LSTM_LAYERS);
bo_all = cell(1, NUM_LSTM_LAYERS);

% Extract weights and biases from each LSTM layer
for i = 1:NUM_LSTM_LAYERS
    % Input gate weights and biases
    Wi_all{i} = all_LSTM_layers{i}.InputWeights(1:NUM_HIDDEN_UNITS, :);
    Ui_all{i} = all_LSTM_layers{i}.RecurrentWeights(1:NUM_HIDDEN_UNITS, :);
    bi_all{i} = all_LSTM_layers{i}.Bias(1:NUM_HIDDEN_UNITS, :);
    
    % Forget gate weights and biases
    Wf_all{i} = all_LSTM_layers{i}.InputWeights(NUM_HIDDEN_UNITS+1:2*NUM_HIDDEN_UNITS, :);
    Uf_all{i} = all_LSTM_layers{i}.RecurrentWeights(NUM_HIDDEN_UNITS+1:2*NUM_HIDDEN_UNITS, :);
    bf_all{i} = all_LSTM_layers{i}.Bias(NUM_HIDDEN_UNITS+1:2*NUM_HIDDEN_UNITS, :);
    
    % Cell state weights and biases
    Wc_all{i} = all_LSTM_layers{i}.InputWeights(2*NUM_HIDDEN_UNITS+1:3*NUM_HIDDEN_UNITS, :);
    Uc_all{i} = all_LSTM_layers{i}.RecurrentWeights(2*NUM_HIDDEN_UNITS+1:3*NUM_HIDDEN_UNITS, :);
    bc_all{i} = all_LSTM_layers{i}.Bias(2*NUM_HIDDEN_UNITS+1:3*NUM_HIDDEN_UNITS, :);
    
    % Output gate weights and biases
    Wo_all{i} = all_LSTM_layers{i}.InputWeights(3*NUM_HIDDEN_UNITS+1:4*NUM_HIDDEN_UNITS, :);
    Uo_all{i} = all_LSTM_layers{i}.RecurrentWeights(3*NUM_HIDDEN_UNITS+1:4*NUM_HIDDEN_UNITS, :);
    bo_all{i} = all_LSTM_layers{i}.Bias(3*NUM_HIDDEN_UNITS+1:4*NUM_HIDDEN_UNITS, :);
end

% % Example of accessing a specific weight (first layer input weights)
% Wi1 = Wi_all{1};

% Display summary of extracted parameters
fprintf('Successfully extracted weights and biases from %d LSTM layers\n', NUM_LSTM_LAYERS);
fprintf('Each layer has %d hidden units\n', NUM_HIDDEN_UNITS);

% Optional: Save the extracted parameters
save('extracted_lstm_parameters.mat', 'Wi_all', 'Ui_all', 'bi_all', 'Wf_all', 'Uf_all', 'bf_all', 'Wc_all', 'Uc_all', 'bc_all', 'Wo_all', 'Uo_all', 'bo_all', 'NUM_HIDDEN_UNITS',"NET_INPUTS","NUM_LSTM_LAYERS");