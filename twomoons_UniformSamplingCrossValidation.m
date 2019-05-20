load('twomoons_sk.mat')

n = length(fea);
% data normalization
feastd = std(fea);
feastd(feastd == 0) = 1; % some pixels are 0 in all images
fea = (fea - mean(fea)) ./ feastd;

% number of sets of samples
n_sets = 1000;
% number of samples in each set
n_samples = 20;
sigma = 0.2; % the sigma is important! default is 1

% final variables
sample_idx = zeros(n_sets, n_samples);
accuracy = zeros(n_sets, 1);

for i = 1:n_sets
	fprintf('sample %d\n', i)
	samples = randsample(n, n_samples);
	A = EuDist2(fea, fea(samples,:));
    
	L = exp(- A ./ (2*sigma^2));
	D1 = sum(L,2);
	D2 = sum(L,1);
	D1 = sparse(1:n,1:n,D1.^(-0.5));
	D2 = sparse(1:n_samples,1:n_samples,D2.^(-0.5));
	L = D1*L*D2;
	[u,s] = svds(L, 2);
	u(:,1) = [];
	u = u ./ vecnorm(u, 2);
	labels = kmeans(u, 2);
	labels = bestMap(gnd, labels);
	acc = sum(labels == gnd) / n;
	accuracy(i) = acc;
	sample_idx(i, :) = samples;
	fprintf('acc %f\n', acc);

end

description = 'twomoons_sk';
save('uniformsampling_accuracy.mat', 'sample_idx', 'accuracy', 'description');
