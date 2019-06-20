function PX = rbfsincos(X, sigma, D)

data_dim = size(X, 2);
r = sigma * randn(data_dim, D);
tmp = X * r;
PX = [cos(tmp); sin(tmp)] ./ sqrt(D);
