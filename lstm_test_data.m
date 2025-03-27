% Generate random data
numSamples = 1000;
a = rand(numSamples, 1);
b = rand(numSamples, 1);
c = rand(numSamples, 1);

% Combine into input matrix
inputs = [a, b, c]';

% Create some outputs (example)
outputs = (a + 2*b - 0.5*c)';

save('lstm_test_data.mat', 'inputs', 'outputs', '-mat');
