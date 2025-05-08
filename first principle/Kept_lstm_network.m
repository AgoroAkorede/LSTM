%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ypred, h1_update,C1_update,h2_update,C2_update] = lstm_network(xt,h1,C1, h2, C2) ;

load('training_data_simple.mat')

net_inputs=1;
num_hidden_units=10 ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First LSTM LAYER
Wu1=pH_net_simple.Layers(2,1).InputWeights ;
WR1=pH_net_simple.Layers(2,1).RecurrentWeights ;
b1=pH_net_simple.Layers(2,1).Bias ;

% SECOND LSTM LAYER
Wu2=pH_net_simple.Layers(3,1).InputWeights ;
WR2=pH_net_simple.Layers(3,1).RecurrentWeights ;
b2=pH_net_simple.Layers(3,1).Bias ;

Wy=pH_net_simple.Layers(4,1).Weights ;
by=pH_net_simple.Layers(4,1).Bias ;

% Define the model weights and biases for each LSTM layer
% First LSTM Layer
Wi1 = Wu1(1:num_hidden_units,:);  % Forget gate input weights (layer 1)
Ui1 = WR1(1:num_hidden_units,:);  % Forget gate recurrent weights (layer 1)
bi1 = pH_net_simple.Layers(2,1).Bias(1:num_hidden_units,:);  % Forget gate biases (layer 1)

Wf1 = pH_net_simple.Layers(2,1).InputWeights(num_hidden_units+1:2*num_hidden_units,:);  % Input gate input weights (layer 1)
Uf1 = pH_net_simple.Layers(2,1).RecurrentWeights(num_hidden_units+1:2*num_hidden_units,:);  % Input gate recurrent weights (layer 1)
bf1 = pH_net_simple.Layers(2,1).Bias(num_hidden_units+1:2*num_hidden_units,:);  % Input gate biases (layer 1)

Wc1 = pH_net_simple.Layers(2,1).InputWeights(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate input weights (layer 1)
Uc1 = pH_net_simple.Layers(2,1).RecurrentWeights(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate recurrent weights (layer 1)
bc1 = pH_net_simple.Layers(2,1).Bias(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate biases (layer 1)

Wo1 = pH_net_simple.Layers(2,1).InputWeights(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state input weights (layer 1)
Uo1 = pH_net_simple.Layers(2,1).RecurrentWeights(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state recurrent weights (layer 1)
bo1 = pH_net_simple.Layers(2,1).Bias(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state biases (layer 1)



% Second LSTM Layer
Wi2 = pH_net_simple.Layers(3,1).InputWeights(1:num_hidden_units,:);  % Forget gate input weights (layer 1)
Ui2 = pH_net_simple.Layers(3,1).RecurrentWeights(1:num_hidden_units,:);  % Forget gate recurrent weights (layer 1)
bi2 = pH_net_simple.Layers(3,1).Bias(1:num_hidden_units,:);  % Forget gate biases (layer 1)

Wf2 = pH_net_simple.Layers(3,1).InputWeights(num_hidden_units+1:2*num_hidden_units,:);  % Input gate input weights (layer 1)
Uf2 = pH_net_simple.Layers(3,1).RecurrentWeights(num_hidden_units+1:2*num_hidden_units,:);  % Input gate recurrent weights (layer 1)
bf2 = pH_net_simple.Layers(3,1).Bias(num_hidden_units+1:2*num_hidden_units,:);  % Input gate biases (layer 1)

Wc2 = pH_net_simple.Layers(3,1).InputWeights(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate input weights (layer 1)
Uc2 = pH_net_simple.Layers(3,1).RecurrentWeights(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate recurrent weights (layer 1)
bc2 = pH_net_simple.Layers(3,1).Bias(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate biases (layer 1)

Wo2 = pH_net_simple.Layers(3,1).InputWeights(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state input weights (layer 1)
Uo2 = pH_net_simple.Layers(3,1).RecurrentWeights(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state recurrent weights (layer 1)
bo2 = pH_net_simple.Layers(3,1).Bias(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state biases (layer 1)


Wy=pH_net_simple.Layers(4,1).Weights;
by=pH_net_simple.Layers(4,1).Bias;

% % Initialize hidden and cell states for both LSTM layers
% h1 = zeros(num_hidden_units, 1);  % Hidden state for layer 1
% C1 = zeros(num_hidden_units, 1);  % Cell state for layer 1
% 
% h2 = zeros(num_hidden_units, 1);  % Hidden state for layer 2
% C2 = zeros(num_hidden_units, 1);  % Cell state for layer 2

% Assuming 'testData' is the input sequence you want to predict on

% testData=test_data_u;
% [num_features, num_time_steps] = size(testData);
% 
% % Initialize an array to store the outputs at each time step
% pH_pred = zeros(net_inputs, num_time_steps);
% outputSequence = zeros(num_hidden_units, num_time_steps);
    
    % ============================
    % First LSTM Layer Calculations
    % ============================
    
    % Forget gate (layer 1)
    f1 = sigmoid(Wf1 * xt + Uf1 * h1 + bf1) ;
    
    % Input gate (layer 1)
    i1 = sigmoid(Wi1 * xt + Ui1 * h1 + bi1);
    
    % Candidate cell state (layer 1)
    C1_candidate = tanh(Wc1 * xt + Uc1 * h1 + bc1);
    
    % Update cell state (layer 1)
    C1 = f1 .* C1 + i1 .* C1_candidate;
    
    % Output gate (layer 1)
    o1 = sigmoid(Wo1 * xt + Uo1 * h1 + bo1);
    
    % Update hidden state (layer 1)
    h1 = o1 .* tanh(C1);
    
    % The output of the first LSTM layer (h1) becomes the input for the second LSTM layer
    
    % ============================
    % Second LSTM Layer Calculations
    % ============================
    
    % Forget gate (layer 2)
    f2 = sigmoid(Wf2 * h1 + Uf2 * h2 + bf2);
    
    % Input gate (layer 2)
    i2 = sigmoid(Wi2 * h1 + Ui2 * h2 + bi2);
    
    % Candidate cell state (layer 2)
    C2_candidate = tanh(Wc2 * h1 + Uc2 * h2 + bc2);
    
    % Update cell state (layer 2)
    C2 = f2 .* C2 + i2 .* C2_candidate;
    
    % Output gate (layer 2)
    o2 = sigmoid(Wo2 * h1 + Uo2 * h2 + bo2);
    
    % Update hidden state (layer 2)
    h2 = o2 .* tanh(C2)  ;

    ypred=Wy*h2+by  ;
    
    % Store the output of the second LSTM layer
    % outputSequence(:, t) = h2;
    % 
    % ypred=Wy*outputSequence+by 

h1_update=h1;
C1_update=C1 ; 
h2_update=h2 ; 
C2_update=C2 ;


% % The final output is the sequence of hidden states from the second LSTM layer
% disp('LSTM Model Output from First Principles (Two LSTM Layers):');
% disp(outputSequence);

% ============================
% Fully Connected Layer Calculation
% ============================

% % For regression, we take the output of the last time step (h2)
% finalHiddenState = h2;  % Output of the last time step from the second LSTM layer
% 
% % Apply fully connected layer
% fcOutput = Wy * finalHiddenState + by;
% 
% % ============================
% % Regression Layer
% % ============================
% % In a regression layer, the output is typically the direct prediction, 
% % so we assume 'fcOutput' is the predicted value
% 
% predictedValue = fcOutput;
% 
% % Display the final predicted value
% disp('Final Predicted Value:');
% disp(predictedValue);

% updatin cell and hidden states



end

% LSTM model prediction ends here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Sigmoid function definition
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end