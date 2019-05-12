load('twomoons_sk.mat')

n = length(fea);
% data normalization
feastd = std(fea);
feastd(feastd == 0) = 1; % some pixels are 0 in all images
fea = (fea - mean(fea)) ./ feastd;

% number of sets of samples
n_sets = 100;
% number of samples in each set
n_samples = 100;

% final table of sigma
sample_idx = zeros(n_sets, n_samples);
accurary = zeros(n_sets, 1);

for i = 1:nsets
	fprintf('sample %d\n', i)
	samples = randsample(n, n_samples);
	A = EuDist2(fea, fea(samples,:));
    
		L = exp(- A ./ (2*sigma^2));
		D1 = sum(L,2);
		D2 = sum(L,1);
		D1 = sparse(1:n,1:n,D1.^(-0.5));
		D2 = sparse(1:n_samples,1:n_samples,D2.^(-0.5));
		L = D1*L*D2;
		[u,s] = svds(L, 3);
		u(:,1) = [];
		u = u ./ vecnorm(u, 2);
		labels = kmeans(u, 2);
		labels = bestMap(gnd, labels);
		acc = sum(labels == gnd) / n_samples;
		table(i,round(sigma)) = acc;
		fprintf('acc %f\n', acc);

end

save('uniformsampling_accuracy.mat', 'sample_idx', 'accuracy');
