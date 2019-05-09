% load('circledata.mat');
clear;
load('dataset/poker.mat')
rng(9999);

n_repeat = 20;
hebb.time = zeros(20, 1);
hebb.acc = zeros(20, 1);
hebb.loss =zeros(20, 7);
svd.time = zeros(20, 1);
svd.acc =zeros(20, 1);
svd.loss = zeros(20, 1);

nlabel = max(gnd);
k = nlabel + 1;
npass = 1;


for i = 1:n_repeat
	[L, idx, D1, D2] = getSparseBipartite(fea, 1000, 30, 'uniform');
	[n,d] = size(L);
	
	[hebb.time(i), hebb.acc(i), hebb.loss(i,:)] = hebbian_run(L, k, nlabel, idx, gnd, D2, n);
	[svd.time(i), svd.acc(i), svd.loss(i,:)] = svd_run(L, k, nlabel, idx, gnd, D2, n);

end


% plot
% figure;
% title('Convergence of Stochastic Gradient Descent');
% xlabel('iteration');
% ylabel('trace(X^TL^TLX)');
% 
% fprintf('hebbian time: %.4f, accuracy: %.4f\n', time.hebb, acc.hebb);
% fprintf('svd time: %.4f, accuracy: %.4f\n', time.svd, acc.svd);
% 
% hold on;
% 
% 
% plot(info_hebb.iter, info_hebb.cost, '.-', 'DisplayName', 'Hebbian');
% plot(info_hebb.iter, bound*ones(size(info_hebb.iter)), '--', 'DisplayName', 'SVD');
% 
% legend('Location', 'SouthEast');
% hold off;

function [time, acc, loss] =  hebbian_run(L, k, nlabel, idx, gnd, D2, n)
	clear opt;
	opt.npass = 1;
	opt.batchsize = 1000;
	opt.stepsize_init = 1e2; % previous 1e2
	opt.stepsize_type = 'decay';
	opt.checkperiod = 10;
	tic;
	[X_hebb, info_hebb] = stochastic_hebbian(L, k, [], opt);
	X_hebb = D2 * X_hebb;
	time = toc;
	label = landmark_cluster(X_hebb, nlabel, idx);
	label = bestMap(gnd, label);
	acc = sum(label == gnd) / n;
	loss = info_hebb.cost;

end



function [time, acc, bound] = svd_run(L, k, nlabel, idx, gnd, D2, n)
	tic;
	[U,S,V] = svds(L, k);
	V = D2 * V;
	time = toc;
	label = landmark_cluster(V, nlabel, idx);
	label = bestMap(gnd, label);
	acc = sum(label == gnd) / n;

	bound = abs(norm(L*V, 'fro')) / n;

end


function label = landmark_cluster(V, nlabel, idx)

[n,~] = size(idx);

V(:,1) = [];
V = V ./ sqrt(sum(V .^2, 2));
reps_labels = litekmeans(V, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates', 10);

label = zeros(n, 1);

for i = 1:n

	label(i) = reps_labels(idx(i));

end

end
