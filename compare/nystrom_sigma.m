load('twomoons_sk.mat')
rng(9999)
n = length(fea);
n_labels = length(unique(gnd));
% data normalization
% feastd = std(fea);
% feastd(feastd == 0) = 1; % some pixels are 0 in all images
% fea = (fea - mean(fea)) ./ feastd;
% number of sets of samples
n_sets = 10;
% number of samples in each set
n_samples = 20;

% sigma list
sigmas = 0.05:0.01:1;
% final table of sigma

table = size(n_sets, length(sigmas));

for i = 1:10
	fprintf('sample %d\n', i)
	samples = randsample(n, n_samples);
	nonsamples = setdiff(1:n, samples);
	W = EuDist2(fea, fea(samples,:), 0);
	gndperm = gnd([samples', nonsamples]);
    
	for sigma = sigmas
		fprintf('sigma %f, ', sigma)
		L = exp(- W ./ (2*sigma^2));
		U = nystrom_approx(L, samples, nonsamples);

        U = U(:,1:n_labels);
		U(:,1) = [];
		U = U ./ vecnorm(U, 2);
		labels = kmeans(U, 2);
		labels = bestMap(gndperm, labels);
		acc = sum(labels == gndperm) / n;

		table(i,round(sigma / 0.01)-4) = acc;
		fprintf('acc %f\n', acc);
	end
end

sigma_acc = mean(table, 1);
save('twomoons_nystrom_sigma.mat', 'table', 'sigma_acc', 'sigmas');
plot(sigma_acc);

function U = nystrom_approx(L, samples, nonsamples)

A = L(samples, :);
B = L(nonsamples, :)';
Asi = sqrtm(pinv(A));

d1 = sum([A, B], 2);
d2 = sum(B', 2) + B' * (pinv(A) * sum(B, 2));

A = A .* d1 .* d1';
B = B .* d1 .* d2';

Q = A + Asi * B * B' * Asi;
[U, S] = svd(Q);

U = [A; B'] * Asi * U * pinv(sqrt(S));


end
