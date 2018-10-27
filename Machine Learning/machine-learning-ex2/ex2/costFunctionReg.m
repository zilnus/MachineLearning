function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% as theta(1) is not calculated in Regularization
% that's why we only calculate theta from 2 to end
R = ((theta(2:end,:)' * theta(2:end,:))) * (lambda / (2 * m));
J = costFunction(theta,X,y) + R;

% gradient start from index 2 will use Regularization
h = sigmoid(X * theta);
grad = (X' * (h - y))/m;
for j=2:n,
  grad(j) = grad(j) + ((theta(j) * lambda)/m);
end

% =============================================================

end
