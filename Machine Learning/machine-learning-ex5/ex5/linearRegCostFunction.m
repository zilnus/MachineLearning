function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% y = 12x1
% X = 12x2
% theta = 2x1

h = X * theta;     % 12x1
dist = ( h' * h ) - ( 2 * (h' * y) ) + (y' * y);   % 1x1

% not calculate theta index 1 in Regularization
R = ( ( theta(2:end,:)' * theta(2:end,:) ) * lambda ) / (2 * m);

% calculate cost function   
J = (dist / (2 * m) ) + R; 

% put theta in temporary variable
% set zero for index 0 as we will use this in gradien descent
% as regularization

temp = theta;
temp(1,:) = 0;

% calculate gradient
grad = ( ( X' * ( h - y ) ) ./ m ) + ( (lambda / m) .* temp);

% =========================================================================

grad = grad(:);

end
