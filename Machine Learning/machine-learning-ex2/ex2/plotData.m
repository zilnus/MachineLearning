function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find which rows in y vector which has value 1
c1 = X(find(y==1),:);
% Find which rows in y vector which has value 0
c2 = X(find(y==0),:);

% Now c1 contains all X1 and X2 for class 1
% Now c2 contains all X1 and X2 for class 0
p1 = plot(c1(:,1), c1(:,2),'k+','LineWidth',2, ...
          'MarkerSize',7);
hold on;
p2 = plot(c2(:,1), c2(:,2),'ko','MarkerFaceColor','y', ...
          'MarkerSize',7);



% =========================================================================



hold off;

end
