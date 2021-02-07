function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% try all combination C and sigma from 8 values constant
% 8^2 = 64 combination

% as we already know the optimal value C and sigma
% then we can skip try all 64 combinations
% speed up submit process
%comb = [0.01 ; 0.03 ; 0.1 ; 0.3 ; 1 ; 3 ; 10 ; 30 ];

% after try 64 combination we get optimal value C and sigma below
combC     = [1];
combsigma = [0.1]; 

min_error = 9999999;
for i=1:1
  C_test = combC(i,:);
  for j=1:1
    sigma_test = combsigma(j,:);
    model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if error < min_error
      min_error = error;
      C = C_test;
      sigma = sigma_test;
    endif
  end
end

% =========================================================================

end
