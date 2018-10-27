function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

%Neular Network

%X = 5000*401
%Theta1 = 25x401
%Theta2 = 10x26


% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
 
% Define hidden layer
hidden = sigmoid(X * Theta1');      % The result will be 5000x25 matrix

% Add bias to hidden layer
hidden = [ones(m,1) hidden];        % Add bias --> 5000x26 matrix

% Define output layer
output = sigmoid(hidden * Theta2'); % The result is 5000*10 matrix

% Define predictions
[max_values,p] = max(output,[],2);

% =========================================================================


end
