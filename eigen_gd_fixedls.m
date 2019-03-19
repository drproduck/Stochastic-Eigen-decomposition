function [X, info] = eigen_gd_fixedls(A,k,opt)

n = size(A,1);

defaults.minstepsize = 0;
defaults.checkperiod = 1;
defaults.maxiter = 100;
defaults.verbosity = 2;
defaults.stepsize_type = 'fixed'
% defaults.linesearch = @linesearch_adaptive;
defaults.tolgradnorm = 1e-15;

if ~exist('opt','var') || isempty(opt)
	opt = struct();

end

opt = mergeOptions(defaults, opt);

opt.maxiter = opt.maxiter - 1;

% Create the problem structure.
manifold = stiefelfactory(n,k);
problem.M = manifold;
 
% Define the problem cost function and its Euclidean gradient.
problem.cost  = @(x) -x'*(A*x);
problem.egrad = @(x) -2*A*x;      % notice the 'e' in 'egrad' for Euclidean

% Numerically check gradient consistency (optional).
% checkgradient(problem);

opt.statsfun = statsfunhelper('metric', @f_cost);

opt.linesearch = @fixed

function cost = f_cost(X)

	cost = trace(X'*A*X);

end


% Solve
[X, xcost, info, opt] = steepestdescent(problem, [], opt);

function [stepsize, newx, newkey, lsstats] = fixed(problem, x, desc_dir, cost, gradnorm, options, storedb, key)
newx = x;
newkey = key;
lsstats = [];
stepsize = 0.01;

end


end

