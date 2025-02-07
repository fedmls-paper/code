% clear variables;
SEEDNUMBER = 0;
rng(SEEDNUMBER, 'twister');
addpath('../tools')

%% Specify the filename
filename = 'breast-cancer-wisconsin.csv';

%% (Optional) Download the dataset if it doesn't exist
if ~isfile(filename)
    % Example URL (update this URL if necessary)
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv';
    fprintf('File "%s" not found. Downloading from %s...\n', filename, url);
    websave(filename, url);
    fprintf('Download complete.\n');
end

%% Load the dataset using readtable (assumes the file has headers)
T = readtable(filename);

%% Inspect the table to verify variable names and sizes
disp(T(1:5,:));

% For this dataset, we assume:
%   - Column 1: 'Id' (ignored)
%   - Columns 2 to 10: Features (numerical)
%   - Column 11: 'Class' (target labels)
features = T{:, 2:10};  % Extract features (columns 2 through 10)
labels = T{:, 11};      % Extract labels from the 'Class' column

%% Convert diagnosis labels to a binary numerical format if needed (M=1, B=0)
if iscell(labels)
    labels_numeric = zeros(size(labels));
    labels_numeric(strcmp(labels, 'M')) = 1;
    labels_numeric(strcmp(labels, 'B')) = 0;
    labels = labels_numeric;
end

%% Split the data into training and test sets
testRatio = 0.2;  % e.g., 20% of the data will be test data
numSamples = size(features, 1);
randIndices = randperm(numSamples);
numTest = round(testRatio * numSamples);
testIdx = randIndices(1:numTest);
trainIdx = randIndices(numTest+1:end);

trainFeatures = features(trainIdx, :);
trainLabels = labels(trainIdx);
trainLabels(trainLabels==0) = -1;  % Convert labels to {-1, +1}
testFeatures = features(testIdx, :);
testLabels = labels(testIdx);
testLabels(testLabels==0) = -1;      % Convert labels to {-1, +1}

fprintf('Training samples: %d, Test samples: %d\n', size(trainFeatures,1), size(testFeatures,1));

%% Impute missing values in the training features (using column means)
for j = 1:size(trainFeatures, 2)
    nanIdx = isnan(trainFeatures(:, j));
    if any(nanIdx)
        colMean = mean(trainFeatures(~nanIdx, j));
        trainFeatures(nanIdx, j) = colMean;
    end
end

%% Optionally, impute missing values in the test features using the training mean
for j = 1:size(testFeatures, 2)
    nanIdx = isnan(testFeatures(:, j));
    if any(nanIdx)
        % Use the mean computed from the training data for consistency
        colMean = mean(trainFeatures(:, j));
        testFeatures(nanIdx, j) = colMean;
    end
end

%% Augment training and test data with a column of ones to incorporate the bias term
trainFeatures_aug = [trainFeatures, ones(size(trainFeatures, 1), 1)];
testFeatures_aug = [testFeatures, ones(size(testFeatures, 1), 1)];

%% Set the number of clients (clusters) for the training data
N = 10;  % Split training data into 10 clients

%% Run k-means clustering on the original (non-augmented) training features
[idx, centroids] = kmeans(trainFeatures, N, 'Replicates', 10);

%% Partition the training data by the cluster assignments
for i = 1:N
    A{i} = trainFeatures_aug(idx == i, :);  % Features for client i (augmented)
    b{i} = trainLabels(idx == i);         % Corresponding labels for client i
    fprintf('Client %d: %d samples\n', i, size(A{i}, 1));
end

%% (Optional) Visualize the clustering result on the training data using the first two features
% figure;
% gscatter(trainFeatures(:,1), trainFeatures(:,2), idx);
% xlabel('Feature 1');
% ylabel('Feature 2');
% title('K-means Clustering of Training Data (Breast Cancer)');

%% Find ground truth by solving the constrained SVM with CVX using the augmented data
[m, d] = size(trainFeatures_aug);  % d is the original feature dimension
R = 1;  % radius for the Euclidean norm ball constraint on the weight part only

cvx_begin
    % xCVX is of size (d) to account for both weights and bias.
    variables xCVX(d)
    % The hinge loss using the augmented training features:
    hinge_loss = max(0, 1 - trainLabels .* (trainFeatures_aug * xCVX));
    minimize( sum(hinge_loss) )
    % Constrain only the original weights, excluding the bias term.
    subject to
        norm(xCVX(1:end-1), 2) <= R % end-1 ensures bias is unconstrained
cvx_end

fprintf('CVX optimal objective: %f\n', cvx_optval);

% Compute Training Accuracy
trainPred = sign(trainFeatures_aug * xCVX);
% Optionally, handle the zero predictions by converting them to 1
trainPred(trainPred == 0) = 1;
trainAccuracy = mean(trainPred == trainLabels);
fprintf('Training Accuracy: %.2f%%\n', trainAccuracy * 100);

% Compute Test Accuracy
testPred = sign(testFeatures_aug * xCVX);
% Handle zero predictions if needed
testPred(testPred == 0) = 1;
testAccuracy = mean(testPred == testLabels);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);

%% Set operators
miniBatchRatio = 0.1;

obj = @(A,b,x) objSVM(A,b,x);
grad = @(A,b,x) gradSVM_stochastic(A,b,x,miniBatchRatio);

nrComRnd = 10000;
nrLcStep0 = 1;

%% FedAvg

LR_Sweep = [1e-4, 1e-3, 1e-2];

xFedAvg = cell(length(LR_Sweep),1);
infoFedAvg = cell(length(LR_Sweep),1);
for i = 1:length(LR_Sweep)
    lr0 = LR_Sweep(i);
    [xFedAvg{i}, infoFedAvg{i}] = FedAvg(A, b, obj, grad, nrComRnd, nrLcStep0, lr0);
end

%% FedMLS

Lambda_Sweep = [1e-1, 1, 10];

xFedMLS = cell(length(Lambda_Sweep),1);
infoFedMLS = cell(length(Lambda_Sweep),1);
for i = 1:length(Lambda_Sweep)
    lambda0 = Lambda_Sweep(i);
    [xFedMLS{i}, infoFedMLS{i}] = FedMLS(A, b, obj, grad, nrComRnd, nrLcStep0, lambda0);
end

%% Save Results

out.xFedAvg = xFedAvg;
out.xFedMLS = xFedMLS;
out.infoFedAvg = infoFedAvg;
out.infoFedMLS = infoFedMLS;
out.xCVX = xCVX;
out.cvx_optval = cvx_optval;

mkdir results;
save(['results/main_svm_stochastic_',num2str(SEEDNUMBER),'.mat'],'out');


%% Scaffnew

% pComm = 1;
% nrIters = infoFedMLS{1}.numLS(end);
% xScaffnew = cell(length(LR_Sweep),1);
% infoScaffnew = cell(length(LR_Sweep),1);
% for i = 1:length(LR_Sweep)
%     lr = LR_Sweep(i);
%     [xScaffnew{i}, infoScaffnew{i}] = Scaffnew(A, b, obj, grad, nrIters, pComm, lr);
% end


%% 
% rel_err = @(val) (val - cvx_optval) ./ cvx_optval;
% 
% figure(1)
% hold off
% for i = 1:length(LR_Sweep)
%     loglog( rel_err(infoFedAvg{i}.obj) ); 
%     hold on;
% end
% 
% figure(2)
% hold off
% for i = 1:length(LR_Sweep)
%     loglog( infoFedAvg{i}.numLS, rel_err(infoFedAvg{i}.obj) ); 
%     hold on;
% end
% 
% figure(3)
% hold off
% for i = 1:length(Lambda_Sweep)
%     loglog( rel_err(infoFedMLS{i}.obj) ); 
%     hold on;
% end
% 
% figure(4)
% hold off
% for i = 1:length(Lambda_Sweep)
%     loglog( infoFedMLS{i}.numLS, rel_err(infoFedMLS{i}.obj) ); 
%     hold on;
% end

% figure(5)
% hold off
% for i = 1:length(LR_Sweep)
%     loglog( rel_err(infoScaffnew{i}.obj) ); 
%     hold on;
% end
% 
% figure(6)
% hold off
% for i = 1:length(LR_Sweep)
%     loglog( infoScaffnew{i}.numLS, rel_err(infoScaffnew{i}.obj) ); 
%     hold on;
% end
% 
