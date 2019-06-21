% load('twomoons_sk.mat')
load('usps.mat')
k = length(unique(gnd));
n = size(fea, 1);
n_sample = 20;
m = n - n_sample;
sigma = 0.2;

samples = randsample(n, n_sample);
W = EuDist2(fea, fea(samples,:));
W = exp(-W / (2*sigma^2));

% nystrom spectral clustering
A = W(samples, :);
nonsamples = setdiff(1:n, samples);
B = W(nonsamples, :)';

% no normalization here
Asi = sqrtm(pinv(A));
Q = A + Asi * B * B' * Asi;

[U, S] = eigs(Q, k);
U = [A; B'] * Asi * U * diag(diag(S) .^ -0.5);
W1 = U*S*U';
D = sum(W1, 2);
da = sparse(1:n_sample,1:n_sample,D(samples).^-0.5);
db = sparse(1:m,1:m,D(nonsamples).^-0.5);
a = da * A * da;
b = da * B * db;


% continue from last
[ua,sa] = eig(A);
sa = diag(sa)'; % row
Ain = ua.*(sa.^-1)*ua;
Asi = ua.*(sa.^-0.5)*ua;

% this degree is correct
d1 = sum([A; B'], 1);
d2 = sum(B, 1) + sum(B', 1) * Ain * B;
dhat = [d1, d2].^(-0.5);

% this noramlization looks correct
A = A.*dhat(1:n_sample)'.*dhat(1:n_sample);
B = B.*dhat(1:n_sample)'.*dhat(n_sample+(1:m));

Q = A + Asi * B * B' * Asi;
[U, S] = svd(Q);
U = [A; B'] * Asi * U * pinv(sqrt(S));

