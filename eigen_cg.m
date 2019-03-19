function [X, v, info] = eigen_cg(A,k)

	n = size(A,1);
	% Create the problem structure.
	manifold = stiefelfactory(n,k);
	problem.M = manifold;
	 
	% Define the problem cost function and its Euclidean gradient.
	problem.cost  = @(x) -x'*(A*x);
	problem.egrad = @(x) -2*A*x;      % notice the 'e' in 'egrad' for Euclidean

	% Numerically check gradient consistency (optional).
	% checkgradient(problem);

	% opt.tolgradnorm = 1e-8;
	% opt.checkperiod = 10;
	opt.maxiter = 1000;
	% opt.batchsize = 100;
	% opt.stepsize_type = 'decay';
	% opt.stepsize_init = 0.1;
	% opt.stepsize_lambda = 1e-3;
	opt.verbosity = 2;
	

	% Solve
	[X, xcost, info, opt] = conjugategradient(problem, [], opt);

	
	[v, s] = eig(A);
	[~, perm] = sort(s, 'descend');
	v = v(:,perm(1:k));
	bound = abs(trace(v'*A*v));

	fprintf('approximate cost: %f\n', trace(X'*A*X));
	fprintf('true cost: %f\n', bound);

	% figure;
	% plot([info.iter], [info.cost], '.-');

	% hold all;
	% plot([info.iter], bound*ones(size([info.iter])), '--');
	% hold off;

end
