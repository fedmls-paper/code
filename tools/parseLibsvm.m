%% Custom function to parse LIBSVM format files
function [Y, X] = parseLibsvm(filename)
    % Open the file for reading
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    % Read the entire file as lines of text
    C = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    lines = C{1};
    nLines = numel(lines);
    
    % Preallocate label vector and storage for sparse matrix data
    Y = zeros(nLines, 1);
    rowIndices = [];
    colIndices = [];
    values = [];
    maxFeature = 0;
    
    % Process each line
    for i = 1:nLines
        line = strtrim(lines{i});
        if isempty(line)
            continue;
        end
        % Each line is of the form: label index1:value1 index2:value2 ...
        tokens = strsplit(line);
        Y(i) = str2double(tokens{1});
        for j = 2:length(tokens)
            kv = strsplit(tokens{j}, ':');
            % Convert feature index and value
            idx = str2double(kv{1});
            val = str2double(kv{2});
            rowIndices(end+1,1) = i;      %#ok<*AGROW>
            colIndices(end+1,1) = idx;
            values(end+1,1) = val;
            maxFeature = max(maxFeature, idx);
        end
    end
    
    % Create the sparse feature matrix
    X = sparse(rowIndices, colIndices, values, nLines, maxFeature);
end