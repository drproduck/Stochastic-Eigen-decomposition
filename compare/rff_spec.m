function U = rff_spec(fea, sigma, fourier_dim)

data_dim = size(fea, 2);
r = sigma * randn(data_dim, fourier_dim);
tmp = fea * r;
W = [cos(tmp), sin(tmp)] ./ sqrt(fourier_dim);
D = W * sum(W', 2);

% approximate Laplacian is D^(-1/2)WWTD(-1/2)
D = sparse(1:n,1:n,D.^(-0.5));
L = D*W;

[U,S] = svds(L, k);
U(:,1) = [];
U = U ./ vecnorm(U, 2);
labels = kmeans(U, 2);
labels = bestMap(gnd, labels);
acc = sum(labels == gnd) / n;
fprintf('accuracy %f\n', acc);

end
