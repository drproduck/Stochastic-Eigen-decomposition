function [label, kept_idx, U, reps] = LBDM(fea, k, opt) 
%LARGE SCALE SPECTRAL CLUSTERING USING DIFFUSION COORDINATE ON
%LANDMARK-BASED BIPARTITE GRAPH

%NOTE TO SELF: currently not multiplying matrix with lbcount

%REQUIRED:
%fea: the data in row-major order (i.e each datapoint is a row)
%k: desired number of clusters
%affinity: currently supports cosine and radial basis function (gaussian)
%r: number of representatives (d. 100)
%s: number of nearest landmarks to keep
%t: diffusion time step
%affinity:
				%'gaussian':
				%'cosine': will normalize features first

%PARAMETER:
%select_method: 
				%'random': pick landmarks uniformly random
				%'++': pick landmarks using kmeans++ weighting
				%'kmeans': pick landmarks as centers of a kmeans run
%embed_method:
				%'landmark': use right singular vector
				%'direct': use left singular vector
				%'coclustering': use both
%cluster_method: algorithms to partition embeddings
				%'kmeans':
				%'discretize:               
% initRes/finalRes: number of restarts for initial/final Kmeans (d. 1/10)
% initIter/finalIter: number of maximum iterations for initial/final

%OPTS 
%sigma: scaling factor for gaussian kernel. Default is computed as
%mean(mean(distance_matrix))

[n,m] = size(fea);

if (~exist('opts','var'))
   opts = [];
end

defaults.r = 100;
defaults.s = 3;
defaults.t = 0;
defaults.affinity = 'gaussian';
defaults.initIter = 10;
defaults.initRes = 1;
defaults.finalIter = 100;
defaults.finalRes = 10;
defaults.select_method = 'kmeans';
defaults.embed_method = 'landmark';
defaults.cluster_method = 'kmeans';
defaults.fid = 1;

if ~exist('opt','var') || isempty(opt)
	opt = struct();

end

opt = mergeOptions(defaults, opt);

r = opt.r;
s = opt.s;
t = opt.t;
affinity = opt.affinity;
initIter = opt.initIter;
initRes = opt.initRes;
finalIter = opt.finalIter;
finalRes = opt.finalRes;
select_method = opt.select_method;
embed_method = opt.embed_method;
cluster_method = opt.cluster_method;
fid = opt.fid;



fprintf('# landmarks = %d\n', r);
fprintf('# nearest landmarks = %d\n', s);
fprintf('# diffusion steps = %d\n', t);
	
%% affinity

%1 cosine

if strcmp(opt.affinity, 'cosine')
	fprintf(fid,'using cosine affinity\n');


	fprintf(fid,'normalizing features...\n');
	fea = fea ./ sqrt(sum(fea.^2, 2));
	
		
	%select landmarks
	fprintf(fid,'selecting landmarks using ');
	tic;
	if strcmp(opt.select_method, 'kmeans') 
		fprintf(fid,'kmeans...\n');
		[lb, reps] = litekmeans(fea, r, 'Distance', 'cosine', 'MaxIter', initIter, 'Replicates', initRes,...
			'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
		lbcount = hist(lb, 1:r); %#ok<NASGU>
	elseif strcmp(select_method, 'uniform')
		fprintf(fid,'random sampling...\n');
		reps = fea(randsample(n, r, false), :);
	else
		error('unsupported mode');
	end
	
	W = fea * reps';
	fprintf(fid,'done in %.2f seconds\n', toc);
	
	%construct A
	fprintf(fid,'constructing sparse A...\n');
	tic;

	if s > 0
		dump = zeros(n,s);
		idx = dump;
		for i = 1:s
			[dump(:,i),idx(:,i)] = max(W,[],2);
			temp = (idx(:,i)-1)*n+(1:n)';
			W(temp) = 1e-100; 
		end
		Gidx = repmat((1:n)',1,s);
		Gjdx = idx;
		W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);

	fprintf(fid,'done in %.2f seconds\n',toc);   

	elseif s <= 0 % default to dense matrix
		fprintf(fid,'Default to dense martix\n');
		if strcmp(embed_method, 'landmark')
			[~,idx] = min(W,[],2);
		end
	end
	   
%2 gaussian
elseif strcmp(affinity, 'gaussian')
	fprintf(fid,'using gaussian affinity\n');
	
	%select landmarks
	fprintf(fid,'selecting landmarks using ');
	tic;

	if strcmp(select_method, 'kmeans')
		fprintf(fid,'kmeans...\n');
%         warning('off', 'stats:kmeans:FailedToConverge')
		[lb, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRes,...
			'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
		lbcount = hist(lb, 1:r);
	
	elseif strcmp(select_method, 'uniform')
		fprintf(fid,'random sampling\n');
		reps = fea(randsample(n, r, false), :);
	   
	elseif strcmp(select_method, '++')
		fprintf(fid,'D2 weight sampling\n');
		[~, reps] = kmeans(fea, r, 'MaxIter',0,'Replicates',1);
	else

		error('unsupported mode');

	end

	fprintf(fid,'done in %.2f seconds\n',toc);
	W = EuDist2(fea, reps, 0);
	
	%determine sigma
	if isfield(opts, 'sigma')
		sigma = opts.sigma;

	elseif strcmp(select_method, 'kmeans')
		sigma = mean(sqrt(VAR ./ lbcount));

	elseif strcmp(select_method, 'uniform') || strcmp(select_method, '++')
		sigma = getSigma(fea);

	end
	
	fprintf(fid,'using sigma = %.2f\n',sigma);
 
	% sparse representation
	fprintf(fid,'constructing sparse A...\n');
	tic;

	if s > 0
		dump = zeros(n,s);
		idx = dump;

		for i = 1:s
			[dump(:,i),idx(:,i)] = min(W,[],2);
			temp = (idx(:,i)-1)*n+(1:n)';
			W(temp) = 1e100; 

		end
		
		%test self-tune sigma
%         sigma = dump(:,s);
%         dump = exp(-dump ./ (2.0 .* sigma .^ 2));

		% manipulate index to efficiently create sparse matrix Z
		% Z is now (normalized to sum 1) smallest r landmarks in each row

        dump = exp(-dump/(2.0*sigma^2));

		Gidx = repmat((1:n)',1,s);
		Gjdx = idx;
		W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);
		
	elseif s <= 0 % default to dense matrix
		fprintf(fid,'default to dense matrix\n');

		if strcmp(embed_method, 'landmark')
			[~,idx] = min(W,[],2);

		end

		W = exp(-W/(2.0*sigma^2));

	end

	fprintf(fid,'done in %.2f seconds\n',toc);

end

%% compute laplacian

fprintf(fid,'Computing Laplacian and diffusion map...\n');

tic;
d1 = sum(W, 2);
d2 = sum(W, 1);
d1 = max(d1, 1e-15);
d2 = max(d2, 1e-15);
D1 = sparse(1:n,1:n,d1.^(-0.5));
D2 = sparse(1:r,1:r,d2.^(-0.5));
L = D1*W*D2;

% expensive svd
[U,S,V] = svds(L, k);

if t > 0

	U = D1 * U * S.^t;
	V = D2 * V * S.^t;

elseif t <= 0

	U = D1 * U;
	V = D2 * V;

end

fprintf(fid,'Done in %.2f seconds\n',toc);

%% cluster embeddings

fprintf(fid,'Clustering result embeddings...\n');
tic;

if strcmp(embed_method, 'landmark')

	if strcmp(cluster_method, 'kmeans')

		V(:,1) = [];
		V = V ./ sqrt(sum(V .^2, 2));
		reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);

	elseif strcmp(cluster_method, 'discretize') 

		reps_labels = discretize(V);

	end

	label = zeros(n, 1);

	for i = 1:n

		label(i) = reps_labels(idx(i));

	end
	
elseif strcmp(embed_method, 'direct')

	if strcmp(cluster_method, 'kmeans')

		U(:,1) = [];
		U = U ./ sqrt(sum(U .^2, 2));
		label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);

	elseif strcmp(cluster_method, 'discretize')

		label = discretize(U);

	end
	
elseif strcmp(embed_method, 'coclustering')

	W = [U;V];

	if strcmp(cluster_method, 'kmeans')

		W(:,1) = [];
		W = W ./ sqrt(sum(W .^2, 2));
		all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);

	elseif strcmp(cluster_method, 'discretize')

		all_label = discretize(W);

	end

	label = all_label(1:n);

end

fprintf(fid,'Done in %.2f seconds\n', toc);


end



