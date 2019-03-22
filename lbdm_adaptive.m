function [X, info] = eigen_adaptivesgd(A,k, X, opt)

defaults.stepsize = 0.001;
defaults.b = 0.999;
defaults.batchsize = 10;
defaults.maxiter = 1000;

if ~exist('opt','var') || isempty(opt)
	opt = struct();

end

opt = mergeOptions(defaults, opt);

% initialize variables
n = size(A,1);
m = 10;
if isempty(X)
	X = randn(n, k);
	[X, ~] = qr(X, 0);

end

l = sparse(n,1);
lh = sparse(n,1);
r = sparse(1,k);
rh = sparse(1,k);

inneriter = ceil(n / opt.batchsize);
outeriter = ceil(opt.maxiter / inneriter);

info.cost_hist = zeros(1, opt.maxiter);
		
for t = 1 : outeriter

	perm = randperm(n);

	for i = 1 : inneriter

		iter = (t - 1)*inneriter + i;

		lower = (i-1)*opt.batchsize+1;	
		upper = min(i*opt.batchsize, n);


		% sparse construct
		sample = perm(lower:upper);
		gg = A(sample,:)*X;
		nsample = length(sample);
		jj = repmat([1:k], nsample, 1);
		ii = repmat(sample, 1, k);
		G = sparse(ii(:), jj(:), gg(:), n, k);
		G = G/nsample;


		G2 = G.^2;
		GGT = sum(G2, 2) ./ k;
		GTG = sum(G2, 1) ./ n; 
		l = opt.b*l + (1-opt.b)*GGT;
		lh = max(lh, l);
		lh = max(lh, 1e-10);
		r = opt.b*r + (1-opt.b)*GTG;
		rh = max(rh, r);
		rh = max(rh, 1e-10);

		L = sparse(1:n, 1:n, l.^(-1/4));
		R = sparse(1:k, 1:k, r.^(-1/4));

		Gadapt = L*G*R;
		PX = Gadapt - X*sym(X'*Gadapt);

		[X,~] = qr(X + opt.stepsize*PX, 0);
		info.cost_hist(iter) = trace(X'*A*X);

		if iter == opt.maxiter
			break;

		end
	
	end

end
		
% options.stepsize_type = 'decay';
% options.stepsize_init = 0.1;
% options.stepsize_lambda = 1e-3;
% options.linesearch = @linesearch_adaptive:


% Plot the special metric recorded by options.statsfun

% Add to that plot a reference: the globally optimal value attained if

% the true dominant singular vectors are computed.

function Y = sym(X)

	Y = (X + X') ./ 2;

end

end
