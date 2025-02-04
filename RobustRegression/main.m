clear variables;
rng(0, 'twister');
addpath('../tools')

%% Download and Load w8a Dataset

% Specify the filename
filename = 'w8a';

% Check if the file already exists in the current folder
if ~isfile(filename)
    % URL for the w8a dataset from the LIBSVM repository
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a';
    fprintf('File "%s" not found. Downloading from %s...\n', filename, url);
    
    % Download the file using websave
    websave(filename, url);
    
    fprintf('Download complete.\n');
end

% Load the dataset
% If you have LIBSVM installed, you can use libsvmread:
if exist('libsvmread', 'file')
    [Y, X] = libsvmread(filename);
    fprintf('Dataset loaded using libsvmread.\n');
else
    % Otherwise, use the custom parser provided below
    [Y, X] = parseLibsvm(filename);
    fprintf('Dataset loaded using the custom LIBSVM parser.\n');
end

[m, d] = size(X);  % number of data samples

% Display some information about the loaded dataset
fprintf('Loaded w8a dataset with %d samples and %d features.\n', size(X,1), size(X,2));

%% Find ground truth by solving with CVX

obj = @(x) norm(X*x - Y,1);

if ~isfile([filename,'_cvx.mat'])
    % Run the CVX optimization only if the file doesn't exist
    cvx_begin
    cvx_precision best
        variable xCVX(d)
        minimize( obj(xCVX) )
    cvx_end

    % Save the result for future use
    save([filename,'_cvx.mat'], 'xCVX');
else
    % Load the precomputed result
    load([filename,'_cvx.mat'], 'xCVX');
end

%% Distribute date accross clients

n = 100;  % for example, you can change this as needed

% Generate a random permutation of the sample indices
indices = randperm(m);

% Preallocate cell arrays to hold the partitioned data
A = cell(n, 1);
b = cell(n, 1);

% Determine roughly how many samples per client
samplesPerClient = floor(m / n);

for i = 1:n
    % Determine the indices for client i
    startIdx = (i - 1) * samplesPerClient + 1;
    endIdx = min(i * samplesPerClient, m);
    clientIndices = indices(startIdx:endIdx);
    
    % Partition the data accordingly
    A{i} = X(clientIndices, :);
    b{i} = Y(clientIndices);
end

%% FedAvg

lr = 0.001;
nrComRnd = 1000;
nrLcStep = 10;
xFedAvg = FedAvg(A, b, nrComRnd, nrLcStep, lr);

%% Scaffnew

lr = 0.001;
nrComRnd = 1000;
pComm = 0.01;
xScaffnew = Scaffnew(A, b, nrComRnd, pComm, lr);

%% FedMLS

G = norm(ones(d,1));
lambda = 10/G^2/2;

nrComRnd = 1000;
% nrLcStep = 10;
% lambda = 100;
xFedMLS = FedMLS(A, b, nrComRnd, lambda, nrLcStep);

%%

obj(xFedAvg)
obj(xScaffnew)
obj(xFedMLS)






