function scores = leverage_score(W, k)

% compute the leverage scores of a matrix relative to the best
% rank-k approximation of A. It is equal to its squared Euclidean
% norms of the rows of the k-svd of A
% A needs to symmetric positive semi definite (?)

% Revisiting the Nystr¨om Method for Improved Large-scale Machine Learning

[U, S] = svds(W, k);
scores = vecnorm(U, 2).^2;
