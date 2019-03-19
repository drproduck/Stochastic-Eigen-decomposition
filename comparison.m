clear all;

simple = 1;
if simple
	load('circledata.mat');
	opt.normalize = 0;
	L = getRBFLaplacian(fea, opt);

else
	load('circledata.mat');
	L = getSparseBipartite(fea, 500, 3);

end

figure;

batchsize = 10;
maxiter = 1000;
stepsize = 0.01;
k = 2;

clear opt;
opt = setopt(batchsize, maxiter, stepsize);
[X_gd, info_gd] = eigen_gd(L, k, opt);

hold on;
size([info_gd.metric])
plot(1:maxiter, [info_gd.metric], 'DisplayName', 'GD');

clear opt;
opt = setopt(batchsize, maxiter, stepsize);
[X_sgd, info_sgd] = eigen_sgd(L, k, opt);
plot(1:maxiter, [info_sgd.metric], 'DisplayName', 'SGD');

clear opt;
opt = setopt(batchsize, maxiter, stepsize);
[X_fast, info_fast] = eigen_fast(L, k, opt);
plot(1:maxiter, info_fast.cost_hist, 'DisplayName', 'Fast');

clear opt;
opt = setopt(batchsize, maxiter, stepsize);
[X_adapt, info_adapt] = eigen_adaptivesgd(L, k, opt);
plot(1:maxiter, info_adapt.cost_hist, 'DisplayName', 'Adaptive');


xlabel('Iteration #');
ylabel('trace(X^T*A*X)');
title('Convergence of stochasticgradient');

% Add to that plot a reference: the globally optimal value attained if

% the true dominant singular vectors are computed.

t = tic();
[V, s] = eig(L);
[~, perm] = sort(diag(s), 'descend');
V = V(:,perm(1:k));
bound = abs(trace(V'*L*V))
fprintf('done: %g [s] (note: svd may be faster)\n', toc(t));

% plot(1:maxiter, bound*ones(maxiter,1), '--');
plot(1:maxiter, bound*ones(maxiter,1), 'DisplayName', 'SVD Bound');
hold off;

legend('Location', 'SouthEast');


function opt = setopt(batchsize, maxiter, stepsize)
	
	opt = struct();

	opt.batchsize = batchsize;
	opt.maxiter = maxiter;
	opt.stepsize = stepsize;

end
