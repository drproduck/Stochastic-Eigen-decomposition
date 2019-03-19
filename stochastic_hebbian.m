function [X, info] = stochastic_hebbian(A,k,X,opt)

% Based on the paper
% Online Generalized Eigenvalue Decomposition: Primal Dual Geometry and Inverse-Free Stochastic Optimization
% http://opt-ml.org/papers/OPT2017_paper_23.pdf
% no orthogonalization

% added adaptive stepsize KHA/et
% Fast Iterative Kernel Principal Component Analysis
% http://www.jmlr.org/papers/volume8/guenter07a/guenter07a.pdf

defaults.checkperiod = 10;
defaults.batchsize = 100;
defaults.npass = 10;
defaults.stepsize_type = 'decay';
defaults.stepsize_init = 0.1;
defaults.stepsize_lambda = 1e-3;
defaults.verbosity = 2;

if ~exist('opt','var') || isempty(opt)
	opt = struct();

end

opt = mergeOptions(defaults, opt);

batchsize = opt.batchsize; 
npass = opt.npass;

% main algorithm
[n,d] = size(A);

if isempty(X)
	X = randn(d,k);

end

info.cost = zeros(npass, 1);
info.iter = [1:npass];

inneriter = ceil(n / batchsize);

for epoch = 1:npass
	
	perm = randperm(n);
	for s = 1:inneriter
		lower = (s - 1)*batchsize+1;
		upper = min(s*batchsize, n);
	
		batch = perm(lower:upper);

		Abatch = A(batch,:);
		G = Abatch*X;
		G = Abatch'*G - X*G'*G;
	
		if strcmp(opt.stepsize_type, 'khaet')
			opt.epoch = epoch;
			eigvec_norm = sqrt(sum(X.^2, 1));
			eigval = zeros(1,k);
			for i = 1:k
				eigval(i) = norm(A*X(:,i), 'fro') / norm(X(:,i), 'fro');

			end

			opt.eigvec_norm = eigvec_norm;
			opt.eigval = eigval;

		end

		global_t = (epoch - 1)*inneriter + s;

		stepsize = getStepsize(global_t, opt);

		X = X + stepsize .* G;

	end

	info.cost(epoch) = norm(A*X, 'fro');
	fprintf('cost = %.4f\n', info.cost(epoch));

end


function stepsize = getStepsize(global_t, opt)

defaults.stepsize_init = 0.1;
defaults.stepsize_type = 'decay';
defaults.stepsize_lambda = 0.1;
defaults.stepsize_decaysteps = 100;

if ~exist('opt', 'var') || isempty(opt)
	options = struct();
end

options = mergeOptions(defaults, opt);


type = opt.stepsize_type;
init = opt.stepsize_init;
lambda = opt.stepsize_lambda;
decaysteps = options.stepsize_decaysteps;


switch type	
	% Step size decays as O(1/iter).
	case 'decay'
		stepsize = init / (1 + init*lambda*global_t);

	case 'khaet'
		stepsize = (opt.eigvec_norm ./ opt.eigval) .* (opt.epoch/(global_t+opt.epoch)*init);
		

	% Step size is fixed.
	case {'fix', 'fixed'}
		stepsize = init;

	% Step size decays only for the few initial iterations.
	case 'hybrid'
		if global_t < decaysteps
			stepsize = init / (1 + init*lambda*global_t);
		else
			stepsize = init / (1 + init*lambda*decaysteps);
		end

	otherwise
		error('step size type not understood');

end

end

end
