function [L, idx] = getBipartite(fea, n_landmarks)

n = size(fea, 1);
idx = randsample(n, n_landmarks);
landmarks = fea(idx, :);


W = pdist2(fea, landmarks, 'squaredeuclidean');

sigma = getSigma(fea);

W = exp(- W / (2*sigma^2));

D1 = sum(W, 2);
D2 = sum(W, 1);

D1 = sparse(1:n,1:n,D1.^(-0.5));
D2 = sparse(1:n_landmarks,1:n_landmarks,D2.^(-0.5));

L = D1 * W * D2;

