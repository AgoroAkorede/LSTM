function prepared_data = prepare_input_data(raw_data, training_settings, seq_length)
    % PREPARE_INPUT_DATA Prepares input data for inference in an NNPC.
    %
    % Inputs:
    %   raw_data: Raw input data (e.g., flow rate values). Should be a numeric array or vector.
    %   training_settings: Normalization settings (e.g., from mapminmax during training).
    %   windowSize: Optional smoothing window size (default = 0 means no smoothing).
    %   seq_length: Length of the input sequence required by the NNPC.
    %
    % Outputs:
    %   prepared_data: Prepared input data ready for inference.

    % Step 1: Extract the most recent sequence of data
    if length(raw_data) < seq_length
        error('Insufficient data points for the required sequence length.');
    end
    recent_sequence = raw_data(end-seq_length+1:end)'; % Extract last `seq_length` points

    % Step 2: Normalize the data
    if ~isempty(training_settings)
        prepared_data = mapminmax('apply', recent_sequence, training_settings);
    else
        error('Normalization settings are required for data preparation.');
    end
    prepared_data = reshape(prepared_data, [size(prepared_data, 1), 1]);
end