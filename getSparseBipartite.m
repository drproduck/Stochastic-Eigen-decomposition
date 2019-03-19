function [L, idx] = getSparseBipartite(fea, r, s)

% LBDM sparse bipatite graph
% Large-scale spectral clustering using diffusion coordinates on landmark-based bipartite graphs
% http://aclweb.org/anthology/W18-1705
% L: the graph
% idx: locations of the nearest landmarks to each datapoint

[n,m] = size(fea);
[lb, reps, ~, VAR] = litekmeans(fea, r,'MaxIter', 10, 'Replicates', 1,...
    'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100,...
    'clustersample', 0.1);
    
%determine sigma
lbcount = hist(lb, 1:r);
cluster_sigma = sqrt(VAR ./ lbcount);
sigma = mean(cluster_sigma); 

n = size(fea, 1);

W = EuDist2(fea, reps, 0);

dump = zeros(n,s);
idx = dump;
for i = 1:s
	[dump(:,i),idx(:,i)] = min(W,[],2);
	temp = (idx(:,i)-1)*n+(1:n)';
	W(temp) = 1e100; 
end

% manipulate index to efficiently create sparse matrix Z
Gidx = repmat((1:n)',1,s);
Gjdx = idx;
W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);

d1 = sum(W, 2);
d1=d1+sum(d1,1)/n;
d2 = sum(W, 1);
d2=d2+sum(d2,2)/r;
D1 = sparse(1:n,1:n,d1.^(-0.5));
D2 = sparse(1:r,1:r,d2.^(-0.5));
L = D1*W*D2;

