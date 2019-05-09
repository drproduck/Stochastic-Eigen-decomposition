function [L, idx, D1, D2, W] = getSparseBipartite(fea, r, s, mode, reps)

% LBDM sparse bipatite graph

% Refs:
% Large-scale spectral clustering using diffusion coordinates on landmark-based bipartite graphs
% http://aclweb.org/anthology/W18-1705

% Args:
% fea: n x d row-based matrix of feature vectors
% r: number of landmarks to sample
% s: number of landmarks distance to keep for each data point. Equivalent to constructing a s-NN
% mode: whether kmeans, uniform sampling, or input should be used to select landmarks
% reps: if mode == 'provided', use input landmarks

% Returns;
% L: the biparite Laplacian matrix = D1^{-1/2} W D2^{-1/2}
% idx: locations of the nearest landmarks to each datapoint
% D1: sparse diagonal matrix of row-sums of W
% D2: sparse diagonal matrix of column-sums of W
% W: the sparse affinity matrix


[n,m] = size(fea);
if strcmp(mode, 'kmeans')
	[lb, reps, ~, VAR] = litekmeans(fea, r,'MaxIter', 10, 'Replicates', 1,...
    'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100,...
    'clustersample', 0.1);

	%determine sigma
	% lbcount = hist(lb, 1:r);
	% cluster_sigma = sqrt(VAR ./ lbcount);
	% sigma = mean(cluster_sigma); 
	sigma = getSigma(fea);
    
elseif strcmp(mode, 'uniform')
	reps = fea(randsample(n, r, false),:);
	sigma = getSigma(fea);

elseif strcmp(mode, 'provided')
	if ~exist('reps', 'var') || isempty(reps) || size(reps, 1) ~= r
			error('invalid landmark points provided');
	else
		% TODO: note here that for sequentially constructing matrix, this sigma is not universal.
		sigma = getSigma(fea);
	end

end


W = EuDist2(fea, reps, 0);

dump = zeros(n,s);
idx = dump;

for i = 1:s
	[dump(:,i),idx(:,i)] = min(W,[],2);
	temp = (idx(:,i)-1)*n+(1:n)';
	W(temp) = 1e100; 
end

% manipulate index to efficiently create sparse matrix Z
dump = exp(-dump / (2.0*sigma^2));
Gidx = repmat((1:n)',1,s);
Gjdx = idx;
W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);

% the regularization is bad for example with mnist. Why?
d1 = sum(W, 2);
% d1=d1+sum(d1,1)/n;
d2 = sum(W, 1);
% d2=d2+sum(d2,2)/r;
D1 = sparse(1:n,1:n,d1.^(-0.5));
D2 = sparse(1:r,1:r,d2.^(-0.5));
L = D1*W*D2;

end
