load('twomoons_sk.mat')
rng(9999)
n = length(fea);
n_labels = length(unique(gnd));
gnd = gnd';
% data normalization
% feastd = std(fea);
% feastd(feastd == 0) = 1; % some pixels are 0 in all images
% fea = (fea - mean(fea)) ./ feastd;
% number of sets of samples
n_sets = 10;
% number of samples in each set
fourier_dim = 100;

% sigma list
sigmas = 0.05:0.01:1;
% final table of sigma

table = size(n_sets, length(sigmas));

for i = 1:10
	fprintf('sample %d\n', i)
    
	for sigma = sigmas
		fprintf('sigma %f, ', sigma)
		U = rff_approx(fea, sigma, fourier_dim);

		U = U(:,1:n_labels);
		U(:,1) = [];
		U = U ./ vecnorm(U, 2);
		labels = kmeans(U, 2);
		labels = bestMap(gnd, labels);
		acc = sum(labels == gnd) / n;

		table(i,round(sigma / 0.01)-4) = acc;
		fprintf('acc %f\n', acc);
	end
end

sigma_acc = mean(table, 1);
% save('twomoons_nystrom_sigma.mat', 'table', 'sigma_acc', 'sigmas');
plot(sigma_acc);

function U = rff_approx(fea, sigma, fourier_dim)

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
