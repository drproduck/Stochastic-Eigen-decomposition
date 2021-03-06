load('Orig.mat')
n = length(fea);
% number of sets of samples
n_sets = 10;
% number of samples in each set
n_samples = 1000;

% sigma list
sigmas = 0.1:0.1:200;
% final table of sigma

table = size(10, length(sigmas));

for i = 1:10
	fprintf('sample %d\n', i)
	samples = randsample(n, n_samples);
	A = EuDist2(fea(samples,:));
	for sigma = sigmas
		fprintf('sigma %f\n', sigma)
		L = exp(- A / (2*sigma^2));
		D = sum(L,2);
		D = sparse(1:n_samples,1:n_samples,D.^(-0.5));
		L = D*L*D;
		[u,s] = svds(L, 11);
		u(:,1) = [];
		u = u ./ vecnorm(u, 2);
		labels = kmeans(u, 10);
		labels = bestMap(gnd(samples),labels);
		acc = sum(labels == gnd(samples)) / n_samples;
		table(i,round(sigma / 0.1)) = acc;
	end
end

sigma_acc = mean(table, 1);
save('mnist_sigma.mat', 'table', 'sigma_acc');
plot(sigma_acc);
