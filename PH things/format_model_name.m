function formatted_name = format_model_name(input_name, include_date)
    % Ensure the input is a valid string or character array
    if ~ischar(input_name) && ~isstring(input_name)
        error('Input must be a string or character array.');
    end
    
    % Convert to lower case and trim spaces
    formatted_name = strtrim(lower(char(input_name)));  % Convert string to char if needed
    
    % Convert to Title Case
    formatted_name(1) = upper(formatted_name(1));  % Capitalize the first letter
    for i = 2:length(formatted_name)
        if formatted_name(i-1) == ' ' || formatted_name(i-1) == '_' % Handle underscores too
            formatted_name(i) = upper(formatted_name(i));
        end
    end
    
    % Remove any special characters except underscores
    formatted_name = regexprep(formatted_name, '[^a-zA-Z0-9_]', '');
    
    % Append the current date if requested
    if include_date
        current_date = datetime('today'); % Get today's date
        formatted_date = datestr(current_date, 'yyyy_mm_dd'); % Format correctly
        formatted_name = [formatted_name, '_', formatted_date];
    end
end