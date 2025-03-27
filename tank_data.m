% Define the parameters

dt=0.1; %sampling time
T = 100; % Total time

t= 0:dt:T;
numSamples = length(t);
h = zeros(numSamples, 3);  % Water levels [h1, h2, h3]
u = rand(numSamples, 3);  % Random pump inputs
A = [0.00875, 0.012075, 0.0035]; % Cross-sectional Area
a= [0.00075,0.0004, 0.0006]; % outlet area
g= 9.81; %Acceleration due to gravity

%% Simulate System Dynamics
for k = 2:numSamples
    h(k,1) = h(k-1,1) + dt * ((u(k,1) - a(1)*sqrt(2*g*h(k-1,1)))/A(1));
    h(k,2) = h(k-1,2) + dt * ((u(k,2) + a(1)*sqrt(2*g*h(k-1,1)) - a(2)*sqrt(2*g*h(k-1,2)))/A(2));
    h(k,3) = h(k-1,3) + dt * ((u(k,3) + a(2)*sqrt(2*g*h(k-1,2)) - a(3)*sqrt(2*g*h(k-1,3)))/A(3));

    % Ensure non-negative heights
    h(k, :) = max(h(k, :), 0);
end

%% Normalize Data for Training
h_min = min(h); h_max = max(h);
h_norm = (h - h_min) ./ (h_max - h_min);
u_min = min(u); u_max = max(u);
u_norm = (u - u_min) ./ (u_max - u_min);

%% Prepare Training Data (Time-Series Format)
seqLength = 5; % Lookback steps

X = {}; Y = {};
for k = seqLength+1:numSamples
    X{end+1} = u_norm(k, :)'; % Past pump inputs (sequence)
    Y{end+1} = h_norm(k, :)'; % Next water level
end

% Convert cell arrays to proper matrices
X = cat(3, X{:}); % Convert cell array to 3D matrix [input_dim, seq_length, num_samples]
Y = cat(2, Y{:}); % Convert cell array to 2D matrix [output_dim, num_samples]

%% Split into Training & Testing Sets (80% Train, 20% Test)
splitRatio = 0.8;
numTrain = round(splitRatio * size(X, 3)); % Get the number of training samples

XTrain = X( :, 1:numTrain);
YTrain = Y(:, 1:numTrain);
XTest = X( :, numTrain+1:end);
YTest = Y(:, numTrain+1:end);

% XTest =X(:,2:numTrain+1);
% YTest =Y(:,2:numTrain+1);


%% Save Data
save('LSTM_Train_data.mat', 'XTrain', 'YTrain', 'XTest', 'YTest', ...
     'h_min', 'h_max', 'u_min', 'u_max');

disp('Data generation complete. Saved to LSTM_Train_data.mat');
