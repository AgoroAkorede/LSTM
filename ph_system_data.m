numSamples = 1000;  % Number of time steps

% Simulated input features (acid/base flow rate, temperature, conductivity)
flow_rate = rand(numSamples, 1) * 10;   % Flow rate (0 to 10 L/min)


% Output variable (pH level)
pH = 7 + 0.5 * sin(0.02 * (1:numSamples)') + 0.1 * randn(numSamples, 1);

% Combine data
Xtrain = flow_rate';
Ytrain = pH';

% Save the data
save('ph_system_data.mat', 'Xtrain', 'Ytrain');
