load('training_data_simple.mat');

[~, q3_init_ss] = get_init_op();
yss = 7;

u_train_un = train_dataset.u.signals(1).values(:,1);
y_train_un = train_dataset.x.signals(3).values(:,1);

xmin=min(u_train_un)  ; xmax=max(u_train_un)  ; 
ymin=min(y_train_un)  ; ymax=max(y_train_un)  ; 

uo = scale_input(q3_init_ss,xmin,xmax,0,1);
u = scale_input(u_train_un,xmin,xmax,0,1);

yo = scale_input(yss,ymin,ymax,0,1);
y = scale_input(y_train_un,ymin,ymax,0,1);

data_size = numel(u);
Ndata = data_size ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_2=[yo , yo, y(1:Ndata-2)'];  
y_1=[yo, y(1:Ndata-1)'];


u_2=[uo , uo,u(1:Ndata-2)'];  
u_1=[uo, u(1:Ndata-1)'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REGRESSED INPUTS AND OUTPUTS FOR 3
X = [y_1 ;  y_2 ; u_1 ;  u_2 ];
Y = y(1:Ndata)';

train_data_u_dim = size(X);
train_data_y_dim = size(Y);

net_inputs = train_data_u_dim(1);
net_outputs = train_data_y_dim(1);
numHiddenUnits = 10;
base_model = "enhanced_ph_model.mat";
model_filename = format_model_name(base_model, true);

% Check for existing model
if exist(model_filename, 'file') == 1
    fprintf('Loading pre-trained model from: %s\n', model_filename);
    load(model_filename);
else
    fprintf('No model found. Starting training .... \n');

layers = [
  sequenceInputLayer(net_inputs)
        lstmLayer(numHiddenUnits, 'OutputMode','sequence', 'StateActivationFunction','tanh', 'GateActivationFunction','sigmoid') 
        dropoutLayer(0.3) % Prevents Overfitting
        lstmLayer(numHiddenUnits)
         lstmLayer(numHiddenUnits)
        fullyConnectedLayer(50)
        reluLayer()
        fullyConnectedLayer(net_outputs)
        regressionLayer
    ];
options = trainingOptions('adam', ...
    'MaxEpochs',20000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',0);

pH_net_simple = trainNetwork(X, Y, layers, options);
save(model_filename, 'pH_net_simple','train_dataset');
  fprintf('New model trained and saved as: %s\n', model_filename);
end
u_test_un = test_dataset.u.signals(1).values(:,1);
y_test_un = test_dataset.x.signals(3).values(:,1);

test_data_size = numel(u_test_un);
Ndata_test = test_data_size ;
u_test = scale_input(u_test_un,xmin,xmax,0,1);
y_test = scale_input(y_test_un,ymin,ymax,0,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_2_test=[yo , yo, y_test(1:Ndata_test-2)'];  
y_1_test=[yo, y_test(1:Ndata_test-1)'];

u_2_test=[uo,uo,u_test(1:Ndata_test-2)'];  
u_1_test=[uo,u_test(1:Ndata_test-1)'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_data_u = [y_1_test; y_2_test ; u_1_test; u_2_test];
test_data_y = y_test(1:end)';


time = test_dataset.u.time';

pred_data_y_unscale = predict(pH_net_simple,test_data_u);

% pred_data_y=scale_input(pred_data_y_unscale,ymin,ymax,0,1) ;
% pred_data_y=scale_input(pred_data_y_unscale,0,1,ymin,ymax) ;
pred_data_y=pred_data_y_unscale ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE MSE for traini process/data
pred_train_data_y = predict(pH_net_simple,X);
mseValue=mean(pred_train_data_y-Y).^2;
disp(['MSE:', num2str(mseValue)])