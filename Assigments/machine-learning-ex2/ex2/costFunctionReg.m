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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

summand_1 = (-y).*log(h);
summand_2 = (-y + 1).*log(-h+1);
grand_summand = summand_1 - summand_2;
J = (1/m) * sum(grand_summand(:)) + ((lambda/(2*m)) * sum(theta(2:size(theta)).^2));

error_vector = h-y;
reg_param = (lambda/m) .* theta(2:size(theta));
grad = (1/m) * (X' * error_vector) + vertcat(0, reg_param);




% =============================================================

end
