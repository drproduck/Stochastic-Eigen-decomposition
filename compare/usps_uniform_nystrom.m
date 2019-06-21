% load('twomoons_sk.mat')
clear;
load('usps.mat');
n_labels = length(unique(gnd));
n = length(fea);
% data normalization
% feastd = std(fea);
% feastd(feastd == 0) = 1; % some pixels are 0 in all images
% fea = (fea - mean(fea)) ./ feastd;

% number of sets of samples
n_sets = 100;
% number of samples in each set
n_samples = 500;
sigma = getSigma(fea);

% final variables
sample_idx = zeros(n_sets, n_samples);

uniform_accuracy = zeros(n_sets, 1);
nystrom_accuracy = zeros(n_sets, 1);

for i = 1:n_sets

	fprintf('sample %d\n', i)
	samples = randsample(n, n_samples);
	nonsamples = setdiff(1:n, samples);
	sample_idx(i, :) = samples;
	W = EuDist2(fea, fea(samples,:), 0);
	W_uniform = exp(- W ./ (2*sigma^2));
	acc = uniform_approx(W_uniform, samples, gnd, n_labels);
	uniform_accuracy(i) = acc;
	fprintf('uniform acc %f\n', acc);

	W_nystrom = exp(- W ./ (2*sigma^2));
	acc = nystrom_approx(W_nystrom, samples, nonsamples, gnd, n_labels);
	nystrom_accuracy(i) = acc;
	fprintf('nystrom acc %f\n', acc);

end

histogram(uniform_accuracy, 100, 'FaceColor', 'b')
hold on
histogram(nystrom_accuracy, 100, 'FaceColor', 'r')
hold off
description = 'twomoons_sk, sigma=0.2, comparison of bipartite and nystrom accuracy for the same samples';
save('twomoons_uniform_vs_nystrom.mat', 'sample_idx', 'uniform_accuracy', 'nystrom_accuracy', 'description');

function acc = uniform_approx(W, samples, gnd, n_labels)

[n,n_samples] = size(W);
D1 = sum(W,2);
D2 = sum(W,1);
D1 = sparse(1:n,1:n,D1.^(-0.5));
D2 = sparse(1:n_samples,1:n_samples,D2.^(-0.5));
L = D1*W*D2;
[u,s] = svds(L, n_labels);
% u(:,1) = [];
u = u ./ vecnorm(u, 2);
labels = kmeans(u, n_labels);
labels = bestMap(gnd, labels);
acc = sum(labels == gnd) / n;

end

function acc = nystrom_approx(W, samples, nonsamples, gnd, n_labels)

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
U = U(:,1:n_labels);
% u(:,1) = [];
U = U ./ vecnorm(U, 2);
labels = kmeans(U, n_labels);
labels = bestMap(gndperm, labels);
acc = sum(labels == gndperm) / n;

end
