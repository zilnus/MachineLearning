function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% X  = 5000x400
% X' = 400x5000
% y  = 5000x1
% y' = 1x5000
% Theta1 = 25x401
% Theta2 = 10x26
% num_labels = 10

% Recode y into values 0 or 1
% if y = 1 then it become 1000000000 (assume num_labels = 10)
% if y = 5 then it become 0000100000 (assume num_labels = 10)
% in this example size Ybinary = 5000x10
Ybinary = zeros(m,num_labels);
for i=1:m
  Ybinary(i,y(i,1)) = 1;
end

% Feedforward

% Add Bias into input layer
X = [ones(m, 1) X];              % X = 5000x401

% Calculate hidden layer
hidden = sigmoid(X * Theta1');   % hidden = 5000x25

% Add Bias into hidden layer
hidden = [ones(m, 1) hidden];    % hidden = 5000x26

% Calculate output layer
output = sigmoid(hidden * Theta2');   % output = 5000x10

% First Way (this is working, without vector operation, just loop on singular operation)
%for i=1:m
%  for(j=1:num_labels)
%    J = J + ( ( -Ybinary(i,j) * log(output(i,j)) ) - ( (1 - Ybinary(i,j)) * log(1 - output(i,j)) ) );  
%  end
%end

% Second Way (this is working, Why?)
% 1x10 * 10x1 produce singular value
%for i=1:m
%  J = J + ( ( -Ybinary(i,:) * log(output(i,:))' ) - ( (1 - Ybinary(i,:)) * log(1 - output(i,:))' ) );
%end

% Third Way (this is not working, why ??? )
% Because 10x1 * 1*10 produce 10x10 matrix value, not a singular value
%for i=1:m
%  J = J + ( ( -Ybinary(i,:)' * log(output(i,:)) ) - ( (1 - Ybinary(i,:)') * log(1 - output(i,:)) ) );
%end

% Fourth Way
cost = ( ( -Ybinary' * log(output) ) - ( (1 - Ybinary') * log(1 - output) ));   % not working, why????
cost = ( ( -Ybinary .* log(output) ) - ( (1 - Ybinary)  .* log(1 - output) ));  % working, why???
J = sum(sum(cost,2));

J = J/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
