clear;
load('twomoons_sk.mat')
gnd = gnd';

n = length(fea);
% data normalization
feastd = std(fea);
feastd(feastd == 0) = 1; % some pixels are 0 in all images
fea = (fea - mean(fea)) ./ feastd;

% number of sets of samples
n_sets = 1000;
% number of samples in each set
n_samples = 20;
sigma = 1.0; % the sigma is important! default is 1

% final variables
sample_idx = zeros(n_sets, n_samples);

uniform_accuracy = zeros(n_sets, 1);
nystrom_accuracy = zeros(n_sets, 1);
rff_accuracy = zeros(n_sets, 1);

W = EuDist2(fea, fea, 0);
W = exp(-W ./ (2*sigma^2));
U_orig = full(W);
for i = 1:n_sets

	fprintf('sample %d\n', i);
	samples = randsample(n, n_samples);
	nonsamples = setdiff(1:n, samples);
	sample_idx(i, :) = samples;
	W = EuDist2(fea, fea(samples,:), 0);

	W_uniform = exp(- W ./ (2*sigma^2));
	U1 = uniform(W_uniform, samples, gnd);
	uniform_accuracy(i) = get_accuracy(U1, gnd);
	dist1 = norm(U_orig'*U1, 'fro') / 2;

	fprintf('uniform acc %f, distance %f\n', uniform_accuracy(i), dist1);

	W_nystrom = exp(- W ./ (2*sigma^2));
	[U2, gndperm] = nystrom(W_nystrom, samples, nonsamples, gnd);
	nystrom_accuracy(i) = get_accuracy(U2, gndperm);
	dist2 = norm(U_orig'*U2, 'fro') / 2;
	fprintf('nystrom acc %f, distance %f\n', nystrom_accuracy(i), dist2);

	U3 = rff(fea, sigma, n_samples);
	rff_accuracy(i) = get_accuracy(U3, gnd);
	dist3 = norm(U_orig'*U3, 'fro') / 2;
	fprintf('random fourier acc %f, distance %f\n', rff_accuracy(i), dist3);

end

histogram(uniform_accuracy, 100, 'FaceColor', 'blue');
hold on
histogram(nystrom_accuracy, 100, 'FaceColor', 'red');
histogram(rff_accuracy, 100, 'FaceColor', 'green');
description = 'twomoons_sk, sigma=0.2, comparison of bipartite and nystrom accuracy for the same samples';
% save('twomoons_uniform_vs_nystrom.mat', 'sample_idx', 'uniform_accuracy', 'nystrom_accuracy', 'description');

function U = uniform(W, samples, gnd)

[n,n_samples] = size(W);
D1 = sum(W,2);
D2 = sum(W,1);
D1 = sparse(1:n,1:n,D1.^(-0.5));
D2 = sparse(1:n_samples,1:n_samples,D2.^(-0.5));
L = D1*W*D2;
[U,S] = svds(L, 2);

end

function [U, gndperm] = nystrom(W, samples, nonsamples, gnd)

n = size(W, 1);
A = W(samples, :);
B = W(nonsamples, :)';
Asi = sqrtm(pinv(A));

d1 = sum([A, B], 2);
d2 = sum(B', 2) + B' * (pinv(A) * sum(B, 2));

A = A .* d1 .* d1';
B = B .* d1 .* d2';

Q = A + Asi * B * B' * Asi;
[U, S] = svd(Q);

U = [A; B'] * Asi * U * pinv(sqrt(S));

gndperm = gnd([samples', nonsamples]);
U = U(:,[1,2]);
% u(:,1) = [];

end

function U = full(W)
n = size(W, 1);
D = sum(W, 2);
D = max(D, 1e-20);
D = sparse(1:n,1:n,D.^(-0.5));
L = D*W*D;

[U,S] = svds(L, 2);

end

function U = rff(fea, sigma, fourier_dim)
n = size(fea, 1);
data_dim = size(fea, 2);
r = sigma * randn(data_dim, fourier_dim);
tmp = fea * r;
W = [cos(tmp), sin(tmp)] ./ sqrt(fourier_dim);
D = W * sum(W', 2);

% approximate Laplacian is D^(-1/2)WWTD(-1/2)
D = sparse(1:n,1:n,D.^(-0.5));
L = D*W;
[U,S] = svds(L, 2);

end

function acc = get_accuracy(U, gnd)

% U(:,1) = [];
n = size(U, 1);
U = U ./ vecnorm(U, 2);
labels = kmeans(U, 2);
labels = bestMap(gnd, labels);
acc = sum(labels == gnd) / n;

end
