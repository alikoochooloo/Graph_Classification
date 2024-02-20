format compact

X = readmatrix('trainfeatures.csv');
Atest = readmatrix('testfeatures.csv');
Y = readmatrix("labels.csv");




%N = size(A, 1);
seed = 10; % seed for random number generator
rng(seed); % for repeatability of your experiment
%A = A(randperm(N),:); % this will reshuffle the rows in matrix A

%X = A(:,1:size(A,2)-1);
%Y = A(:,size(A,2));

confusion = [0,0;0,0];
total_error = 0;

limit = floor(N/10);

for i = 1:10
    lower = i * limit - limit + 1;
    upper = i * limit;
    Xtrain = cat(1,X(1:lower-1,:), X(upper+1:size(X,1),:));
    Ytrain = cat(1,Y(1:lower-1), Y(upper+1:size(Y)));
    Xtest = X(lower:upper,:);
    Ytest = Y(lower:upper);
    
    lambda = logspace(-4,3,11); % create a set of candidate lambda values
    SVMmodel = fitcecoc(Xtrain, Ytrain, 'Kfold', 5, 'Learner', 'svm', 'Lambda', lambda);
    %foldNumber = 3; % to examine the model created for the 3rd fold
    %SVMmodel.Trained{foldNumber}
    ce = kfoldLoss(SVMmodel); % to examine the classification error for each lambda
    bestIdx = find(ce == min(ce)); % identify the index of lambda with smallest error
    %bestIdx
    bestLambda = lambda(bestIdx(1));

    SVMmodel = fitcecoc(Xtrain, Ytrain, 'Learner', 'svm', 'Lambda', bestLambda);
    pred = predict(SVMmodel, Xtest);
    
    cp = classperf(Ytest);
    classperf(cp,pred);
    cp.DiagnosticTable % to show the confusion matrix
    cp.ErrorRate % to show the classification error
    
    confusion = confusion + cp.DiagnosticTable;
    total_error = total_error + cp.ErrorRate;
end

confusion
total_error = total_error/10
