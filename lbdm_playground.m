% load('circledata.mat');
clear;
load('Orig.mat')
rng(1234);

[L, idx] = getSparseBipartite(fea, 1000, 3);
[n,d] = size(L);
k = 10;
npass = 10;

% shared starting point
X = randn(d,k);
X = X ./ sqrt(sum(X.^2, 1));

% clear opt;
% opt.npass = 20;
% opt.batchsize = 1000;
% opt.stepsize_init = 1e2;
% opt.stepsize_type = 'khaet';
% tic;
% [X_khaet, info_khaet] = stochastic_hebbian(L, k, X, opt);
% time.khaet = toc;
% label = landmark_cluster(X_khaet, k, idx);
% label = bestMap(gnd, label);
% acc.khaet = sum(label == gnd) / n;


clear opt;
opt.npass = 10;
opt.batchsize = 1000;
opt.stepsize_init = 1e2;
opt.stepsize_type = 'decay';
tic;
[X_hebb, info_hebb] = stochastic_hebbian(L, k, X, opt);
time.hebb = toc;
label = landmark_cluster(X_hebb, k, idx);
label = bestMap(gnd, label);
acc.hebb = sum(label == gnd) / n;


clear opt
opt.batchsize = 1000;
opt.stepsize_init = 0.01;
opt.npass = 10;
opt.stepsize_init = 1e2;
tic;
[X_sgd, info_sgd] = lbdm_sgd(L, k, X, opt);
time.sgd = toc;
label = landmark_cluster(X_sgd, k, idx);
label = bestMap(gnd, label);
acc.sgd = sum(label == gnd) / n;


% clear opt;
% opt.maxiter = 1000;
% tic;
% [X_gd, info_gd] = lbdm_gd(L, k, X, opt);
% time.gd = toc;
% label = landmark_cluster(X_gd, k, idx);
% label = bestMap(gnd, label);
% acc.gd = sum(label == gnd) / n;


tic;
[U,S,V] = svds(L);
time.svd = toc;
label = landmark_cluster(V, k, idx);
label = bestMap(gnd, label);
acc.svd = sum(label == gnd) / n;

bound = abs(norm(L*V, 'fro'));


% plot
figure;
title('Convergence of Stochastic Gradient Descent');
xlim([1,10]);
xlabel('# passes through data');
ylabel('trace(X^TL^TLX)');
% fprintf('hebbian khaet time: %.4f, accuracy: %.4f\n', time.khaet, acc.khaet);
fprintf('hebbian time: %.4f, accuracy: %.4f\n', time.hebb, acc.hebb);
fprintf('sgd time: %.4f, accuracy: %.4f\n', time.sgd, acc.sgd);
% fprintf('gd time: %.4f, accuracy: %.4f\n', time.gd, acc.gd);
fprintf('svd time: %.4f, accuracy: %.4f\n', time.svd, acc.svd);

hold on;
% plot(info_khaet.iter, info_khaet.cost, '.-', 'DisplayName', 'Hebbian khaet');

plot(info_hebb.iter, info_hebb.cost, '.-', 'DisplayName', 'Hebbian');

% reduce gd result
% gd_resl = length([info_gd.metric]);
% interv = floor(gd_resl / 20);
% gd_metric = fliplr([info_gd.metric]);
% gd_metric = gd_metric(1:interv:length(gd_metric));
% gd_metric = fliplr(gd_metric);
% plot(1:length(gd_metric), gd_metric, '.-', 'DisplayName', 'GD');

sgd_metric = [info_sgd.metric];
sgd_metric(end) = [];
plot(1:length(sgd_metric), sgd_metric, '.-', 'DisplayName', 'R-SGD');

plot(1:length([info_sgd.metric]), bound*ones(size([info_sgd.metric])), '--', 'DisplayName', 'SVD');

legend('Location', 'SouthEast');
hold off;


function label = landmark_cluster(V, k, idx)

[n,~] = size(idx);

V(:,1) = [];
V = V ./ sqrt(sum(V .^2, 2));
reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates', 10);

label = zeros(n, 1);

for i = 1:n

	label(i) = reps_labels(idx(i));

end

end
