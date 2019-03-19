function [X, info] = eigen_fast(A, k, opt)

% if isempty(A)
% 	fprintf('matrix not given. Default to random matrix of size 1000\n')
% 
% 	A = randn(1000, 1000);
% 	A = (A + A') / 2;

% end

n = size(A, 1);

defaults.stepsize = 0.01;
defaults.maxiter = 1000;
defaults.batchsize = 100;

if ~exist('opt','var') || isempty(opt)
	opt = struct();

end

opt = mergeOptions(defaults, opt);

X = randn(n, k);
X = X ./ sqrt(sum(X.^2, 1));

inneriter = ceil(n / opt.batchsize);
outeriter = ceil(opt.maxiter / inneriter);

info.cost_hist = zeros(1, opt.maxiter);

for s = 1:outeriter

	L = A*X;
	Gouter = L - X*X'*L;
	Xinner = X;

	perm_idx = randperm(n);

	for t = 1:inneriter

		iter = (s - 1)*inneriter + t;

		lower = opt.batchsize*(t-1)+1;
		upper = opt.batchsize*t;

		if upper > n
			upper = n;

		end

		sample = perm_idx(lower:upper);
		nsample = length(sample);

		ii = repmat(sample', 1, n);
		jj = repmat(1:n, nsample, 1);
		aa = A(sample, :);
		Asample = sparse(ii(:), jj(:), aa(:), n, n);
		Gouter_batch = (speye(n) - X*X')*(Asample*X);

		[p1,~,p2] = svd(Xinner'*X);

		Gvariate = (Gouter_batch - Gouter)*p2'*p1;

		Gvariate_trans = Gvariate - Xinner*sym(Xinner'*Gvariate);

		Ginner_batch = (speye(n) - Xinner*Xinner')*(Asample*Xinner);

		Greduce = Ginner_batch - Gvariate_trans;

		Xtangent = Xinner + opt.stepsize*Greduce;

		Xinner = Xtangent*(Xtangent'*Xtangent)^(-0.5);
		% sqrt(sum(Xinner.^2, 1))

		info.cost_hist(iter) = trace(Xinner'*A*Xinner);

		% fprintf('outer loop %d, inner loop %d, cost=%f\n', s, t, cost_hist((s-1)*inneriter+t));

		if iter == opt.maxiter
			break

		end

	end

	X = Xinner;

end


function Y = sym(X)

Y = (X + X') ./ 2;

end

end
