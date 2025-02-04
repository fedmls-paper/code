clear variables;
rng(0, 'twister');

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

X = X(1:1000,:);
Y = Y(1:1000);

filename = [filename, '_toy'];

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

N = 10;  % for example, you can change this as needed

% Generate a random permutation of the sample indices
indices = randperm(m);

% Preallocate cell arrays to hold the partitioned data
A = cell(N, 1);
b = cell(N, 1);

% Determine roughly how many samples per client
samplesPerClient = floor(m / N);

for i = 1:N
    % Determine the indices for client i
    startIdx = (i - 1) * samplesPerClient + 1;
    endIdx = min(i * samplesPerClient, m);
    clientIndices = indices(startIdx:endIdx);
    
    % Partition the data accordingly
    A{i} = X(clientIndices, :);
    b{i} = Y(clientIndices);
end

%% FedAvg

lr = 0.01;
nrComRnd = 10000;
nrLcStep0 = 0.1;
[xFedAvg, infoFedAvg] = FedAvg(A, b, nrComRnd, nrLcStep0, lr);

%% Scaffnew

lr = 0.01;
nrComRnd = 10000000;
pComm = 1;
[xScaffnew, infoScaffnew] = Scaffnew(A, b, nrComRnd, pComm, lr);

%% FedMLS

G = norm(ones(d,1));
lambda0 = 100/G^2;
nrLcStep0 = 2*lambda0;
nrComRnd = 10000;

[xFedMLS, infoFedMLS] = FedMLS(A, b, nrComRnd, nrLcStep0, lambda0);

%%

obj(xFedAvg)
obj(xScaffnew)
obj(xFedMLS)

%%

cvx_optval = obj(xCVX);
rel_err = @(val) (val - cvx_optval) ./ cvx_optval;

close all
figure(1)
loglog(rel_err(infoFedMLS.obj), 'b')
hold on
loglog(rel_err(infoFedAvg.obj), 'r')
loglog(rel_err(infoScaffnew.obj), 'k')
loglog(infoFedMLS.numLS ,rel_err(infoFedMLS.obj), 'b--')
loglog(infoFedAvg.numLS ,rel_err(infoFedAvg.obj), 'r--')
loglog(infoScaffnew.numLS ,rel_err(infoScaffnew.obj), 'k--')
grid on






