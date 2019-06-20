load('twomoons_sk.mat')

n = length(fea);
% data normalization
% feastd = std(fea);
% feastd(feastd == 0) = 1; % some pixels are 0 in all images
% fea = (fea - mean(fea)) ./ feastd;

% number of sets of samples
n_sets = 1000;
% number of samples in each set
n_samples = 20;
sigma = 0.2; % the sigma is important! default is 1

% final variables
sample_idx = zeros(n_sets, n_samples);

uniform_accuracy = zeros(n_sets, 1);
nystrom_accuracy = zeros(n_sets, 1);

W = EuDist2(fea, fea, 0);
W = exp(-W ./ (2*sigma^2));
U_orig = full(W);
for i = 1:n_sets

	fprintf('sample %d\n', i)
	samples = randsample(n, n_samples);
	nonsamples = setdiff(1:n, samples);
	sample_idx(i, :) = samples;
	W = EuDist2(fea, fea(samples,:), 0);

	W_uniform = exp(- W ./ (2*sigma^2));
	U1 = uniform(W_uniform, samples, gnd);
	uniform_accuracy(i) = norm(U_orig'*U1, 'fro') / 2;
	fprintf('uniform acc %f\n', uniform_accuracy(i));

	W_nystrom = exp(- W ./ (2*sigma^2));
	U2 = nystrom(W_nystrom, samples, nonsamples, gnd);
	nystrom_accuracy(i) = norm(U_orig'*U2, 'fro') / 2;
	fprintf('nystrom acc %f\n', nystrom_accuracy(i));

end

histogram(uniform_accuracy, 100, 'FaceColor', 'blue');
hold on
histogram(nystrom_accuracy, 100, 'FaceColor', 'red');
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

function U = nystrom(W, samples, nonsamples, gnd)

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
