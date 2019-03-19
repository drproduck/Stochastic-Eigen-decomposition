function [X, info] = eigen_gd(A,k,opt)

n = size(A,1);

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

	cost = trace(X'*A*X);

end


% Solve
[X, xcost, info, opt] = steepestdescent(problem, [], opt);

end

