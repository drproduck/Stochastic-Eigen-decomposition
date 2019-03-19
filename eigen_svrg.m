function [x, infos] = eigen_svrg(A,k,opt)

maxepoch = 360; % outer loop`
n = size(A,1); % the size of matrix
tolgradnorm = 1e-8; % stop when gradient norm is below this
batchsize = 10; % # samples in inner loop, default 1
inner_repeat = 1;
srg_varpi = 0.5;

% set manifold
problem.M = stiefelfactory(n, k);
problem.ncostterms = n; % as per stochastic_PCA

% set problem
% set cost. recall the minimization problem is -(1/2) (1/n) Tr(x'Ax) s.t x'x = I_p
problem.cost = @cost;
function f = cost(x)
	f = trace(x'*A*x);
end

% set full gradient
problem.egrad = @egrad;
function g = egrad(x)
	g = -A*x;
	g = g./n;
end

% set partial gradient
problem.partialegrad = @partialegrad;
function g = partialegrad(x, sample)
	gg = -A(sample,:)*x;
	nsample = length(sample);
	jj = repmat([1:k], nsample, 1);
	ii = repmat(sample, 1, k);
	g = sparse(ii(:), jj(:), gg(:), n, k);
	g = g/nsample;
	
end

% initialize
Uinit = problem.M.rand();

% options
clear opt;
opt.verbosity = 1;
opt.batchsize = 10;
opt.update_type = 'svrg';
opt.maxepoch = 30;
opt.tolgradnorm = tolgradnorm;
opt.svrg_type = 1;
opt.stepsize_type = 'fix';
opt.stepsize = 0.01; % default 1e-6
opt.boost = 0;
opt.transport = 'ret_vector_locking';

[x, xcost, infos, options] = Riemannian_svrg(problem, Uinit, opt);

[v,s] = eig(A);
[~,p] = sort(s,'descend');
v = v(:,p(1:k));

bound = abs(trace(v'*A*v));	
infos

figure;
plot([infos.epoch], bound*ones(size([infos.epoch])), '--', 'DisplayName', 'eigs bound');
hold on
plot([infos.epoch], [infos.cost], '.-', 'DisplayName', 'svrg');
hold off
legend('Location', 'SouthEast');

end

