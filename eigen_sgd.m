function [X, info] = eigen_sgd(A,k,opt)

% stochastic riemannian
n = size(A,1);
problem.M = stiefelfactory(n,k);
m = 10;

problem.ncostterms = n; % see stochastic_pca
problem.cost = @(X) trace(X'*A*X);

problem.partialegrad = @partialegrad;
function G = partialegrad(X, sample)
	% % compute partial gradient
	% nsample = length(sample);
	% Asample = zeros(n,n); % for i = 1:nsample
	% 	Asample(sample(i),:) = A(sample(i),:);
	% end
	% G = -(1/nsample)*(Asample*X);
	
	% sparse construct
	gg = -A(sample,:)*X;
	nsample = length(sample);
	jj = repmat([1:k], nsample, 1);
	ii = repmat(sample, 1, k);
	G = sparse(ii(:), jj(:), gg(:), n, k);
	G = G/nsample;
	
end

opt.checkperiod = 1;
% options.batchsize = 100;
opt.maxiter = opt.maxiter - 1;
opt.stepsize_type = 'decay';
opt.stepsize_init = 0.1;
opt.stepsize_lambda = 1e-3;
opt.verbosity = 2;
% options.linesearch = @linesearch_adaptive:
opt.statsfun = statsfunhelper('metric', @(X) trace(X'*A*X));

[X, info] = stochasticgradient(problem, [], opt);


end
