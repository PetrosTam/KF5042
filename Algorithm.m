clc
close all
clearvars

%% Step 1: Dataset Preparation
% Prepare the data for SVM input by loading it from a CSV file, extracting necessary columns, and shuffling it to avoid variance or bias issues.
%%
optionsSet = detectImportOptions('BCCD.csv');
fileData = readtable('BCCD.csv', optionsSet);
cellData = table2cell(fileData);

numSamples = size(cellData, 1);

% Create X and Y for SVM training
randNums = randperm(numSamples);

% Load X with features
X_temp = cell2mat(cellData(:, 1:9));
X = X_temp(randNums(1:end), :);

numFeatures = size(X, 2);

% Load Y with labels
Y_temp = cell2mat(cellData(:, 10));
Y = Y_temp(randNums(1:end), :);

%% Step 2: Perform Cross Validation using K-fold 10 times
% Cross-validate the data using K-Fold 10 times with stratified folds to reduce underfitting or overfitting and lower data bias.
%%
CV_set = cvpartition(Y, 'KFold', 10, 'Stratify', true);

%% Step 3: Feature Ranking
% Rank the features and create feature sets ranging from 1 feature to all features of the dataset.
%%
optionsSet = statset('display', 'iter', 'UseParallel', true);
rng(5);

modelFunc = @(trainData, trainLabels, testData, testLabels)...
    sum(predict(fitcsvm(trainData, trainLabels, 'Standardize', true, 'KernelFunction', 'gaussian'), testData) ~= testLabels);

% Rank the features using sequential forward selection
[featSelect, history] = sequentialfs(modelFunc, X, Y, 'cv', CV_set, ...
    'options', optionsSet, 'nfeatures', numFeatures);

%% Step 4: Kernel and Feature Selection
% Analyze the performance of 3 kernel functions for each feature set.
%%
rng(3);
PerfMatrix(numFeatures, 6) = 0;
for count = 1:numFeatures
    PerfMatrix(count, 1) = count;
    
    % Linear
    linearModel = fitcsvm(X(:, history.In(count, :)), Y, 'BoxConstraint', 1, 'CVPartition', CV_set, 'KernelFunction', ...
        'linear', 'Standardize', true, 'KernelScale', 'auto');
   
    PerfMatrix(count, 2) = (1 - kfoldLoss(linearModel, 'LossFun', 'ClassifError')) * 100;
    
    % Polynomial
    polyModel = fitcsvm(X(:, history.In(count, :)), Y, 'BoxConstraint', 1, 'CVPartition', CV_set, 'KernelFunction', ...
        'polynomial', 'Standardize', true, 'KernelScale', 'auto');
   
    PerfMatrix(count, 3) = (1 - kfoldLoss(polyModel, 'LossFun', 'ClassifError')) * 100;
    
    % Gaussian
    gaussianModel = fitcsvm(X(:, history.In(count, :)), Y, 'BoxConstraint', 1, 'CVPartition', CV_set, 'KernelFunction', ...
        'gaussian', 'Standardize', true, 'KernelScale', 'auto');
   
    PerfMatrix(count, 4) = (1 - kfoldLoss(gaussianModel, 'LossFun', 'ClassifError')) * 100;
end

%% Visualize Results
%%
figure
plot(PerfMatrix(:, 2:6))
title('Breast Cancer Coimbra Model Performance')
xlabel('Number of Ranked Features')
ylabel('Model Performance(%)')
legend('Linear', 'Polynomial', 'Gaussian')
grid on;

%% Step 5: Select the best kernel function
% Select the best hyperparameters for the highest accuracy performance for the dataset.
%%
rng(3);

% The best observed Kernel is Gaussian
gaussianModel = fitcsvm(X(:, history.In(5, :)), Y, 'KernelFunction', ...
    'gaussian', 'Standardize', true, 'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('UseParallel', true, ...
    'ShowPlots', false, 'MaxObjectiveEvaluations', 80, 'Repartition', true));

%% Step 6: Train the SVM model, find the Accuracy, and evaluate the validation loss
%%
rng(3);

boxConst = gaussianModel.BoxConstraints(1, :);
kernelScale = gaussianModel.KernelParameters.Scale;

% Using the Gaussian kernel
optimalModel = fitcsvm(X(:, history.In(4, :)), Y, 'CVPartition', CV_set, 'KernelFunction', ...
    'gaussian', 'Standardize', true, 'BoxConstraint', boxConst, 'KernelScale', kernelScale);

% Compute validation accuracy
ValidationAccuracy = (1 - kfoldLoss(optimalModel, 'LossFun', 'ClassifError')) * 100
% Compute validation loss
ValidationLoss = kfoldLoss(optimalModel) * 100

%% Step 7: Evaluate the model's performance using a confusion matrix
figure
[Y_pred, valScores] = kfoldPredict(optimalModel);
confMat = confusionmat(Y, Y_pred);

% Create a confusion matrix
conMatHeatmap = heatmap(confMat, 'Title', 'Confusion Matrix of BCCD dataset', 'YLabel', 'True Diagnosis', 'XLabel', 'Predicted Diagnosis', ...
    'XDisplayLabels', {'Healthy(1)', 'Patients(2)'}, 'YDisplayLabels', {'Healthy(1)', 'Patients(2)'}, 'ColorbarVisible', 'off');

% Define functions to calculate recall, specificity, precision, and F1 Score
recallFunc = @(confMatrix) diag(confMatrix) ./ sum(confMatrix, 2);
specificityFunc = @(confMatrix) diag(confMatrix) ./ sum(confMatrix, 2);
precisionFunc = @(confMatrix) diag(confMatrix) ./ sum(confMatrix, 1);
f1ScoreFunc = @(confMatrix) 2 * (precisionFunc(confMatrix) .* recallFunc(confMatrix)) ./ (precisionFunc(confMatrix) + recallFunc(confMatrix));

Recall = recallFunc(confMat) * 100
Specificity = specificityFunc(confMat) * 100
Precision = precisionFunc(confMat) * 100
F1Score = f1ScoreFunc(confMat) * 100