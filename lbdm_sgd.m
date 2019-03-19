function [X, info] = lbdm_sgd(A,k,X,opt)

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

% stochastic riemannian
[n,d] = size(A);

if isempty(X)
	X = randn(d, k);
	X = X ./ sqrt(sum(X.^2, 1));

end

problem.M = stiefelfactory(d,k);

problem.ncostterms = n; % see stochastic_pca

problem.partialegrad = @partialegrad;
function G = partialegrad(X, sample)
	Asample = A(sample,:);
	G = -Asample'*(Asample*X);
	% G = G/ n;
	
end

opt.maxiter = ceil(n / opt.batchsize) * opt.npass;
opt.checkperiod = ceil(n / opt.batchsize);
opt.linesearch = @linesearch_adaptive;
opt.statsfun = statsfunhelper('metric', @(X) norm(A*X, 'fro'));

[X, info] = stochasticgradient(problem, X, opt);


end
