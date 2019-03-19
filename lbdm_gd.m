function [X, info] = lbdm_gd(L,k,X,opt)

A = L'*L;
[n,d] = size(A);

if isempty(X)
	X = randn(d, k);
	X = X ./ sqrt(sum(X.^2, 1));

end

defaults.minstepsize = 0;
defaults.checkperiod = 1;
defaults.maxiter = 100;
defaults.verbosity = 2;
defaults.linesearch = @linesearch_adaptive;
defaults.tolgradnorm = 1e-15;

if ~exist('opt','var') || isempty(opt)
	opt = struct();

end

opt = mergeOptions(defaults, opt);

% Create the problem structure.
manifold = stiefelfactory(n,k);
problem.M = manifold;
 
% Define the problem cost function and its Euclidean gradient.
problem.cost  = @(x) -x'*(A*x);
problem.egrad = @(x) -2*A*x;      % notice the 'e' in 'egrad' for Euclidean

% Numerically check gradient consistency (optional).
% checkgradient(problem);

opt.statsfun = statsfunhelper('metric', @f_cost);

function cost = f_cost(X)

	cost = norm(L*X, 'fro');

end


% Solve
[X, xcost, info, opt] = steepestdescent(problem, [], opt);

end

