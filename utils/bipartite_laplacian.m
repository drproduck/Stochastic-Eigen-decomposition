function L = bipartite_laplacian(W)

[n1, n2] = size(W);

D1 = sum(W, 2);
D2 = sum(W, 1);

% try to avoid underflow
D1 = max(D1, 1e-100);
D2 = max(D2, 1e-100);

D1 = sparse(1:n1,1:n1,D1.^(-0.5));
D2 = sparse(1:n2,1:n2,D2.^(-0.5));

L = D1 * W * D2;

