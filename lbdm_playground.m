% load('circledata.mat');
clear;
load('Orig.mat')
% rng(9999);

[L, idx, D1, D2] = getSparseBipartite(fea, 1000, 3, 'uniform');
% fprintf('constructing full graph\n')
% tic;
% [L, idx] = getBipartite(fea, 1000, 3, 'uniform');
% fprintf('done in %f seconds\n', toc)

[n,d] = size(L);
k = 11;
nlabel = 10;
npass = 10;

% shared starting point
X_0 = randn(d,k);
[X_0,~] = qr(X_0, 0);

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




% clear opt
% opt.npass = 5;
% opt.batchsize = 1000;
% opt.stepsize_init = 1e2;
% tic;
% [X_sgd, info_sgd] = lbdm_sgd(L, k, X_0, opt);
% time.sgd = toc;
% label = landmark_cluster(X_sgd, nlabel, idx);
% label = bestMap(gnd, label);
% acc.sgd = sum(label == gnd) / n;


clear opt;
opt.npass = 1;
opt.batchsize = 1000;
opt.stepsize_init = 1e2; % previous 1e2
opt.stepsize_type = 'decay';
opt.checkperiod = 10;
tic;
[X_hebb, info_hebb] = stochastic_hebbian(L, k, X_0, opt);
time.hebb = toc;
label = landmark_cluster(X_hebb, nlabel, idx);
label = bestMap(gnd, label);
acc.hebb = sum(label == gnd) / n;


% clear opt;
% opt.maxiter = 1000;
% tic;
% [X_gd, info_gd] = lbdm_gd(L, k, X, opt);
% time.gd = toc;
% label = landmark_cluster(X_gd, k, idx);
% label = bestMap(gnd, label);
% acc.gd = sum(label == gnd) / n;


tic;
[U,S,V] = svds(L, k);
time.svd = toc;
label = landmark_cluster(V, nlabel, idx);
label = bestMap(gnd, label);
acc.svd = sum(label == gnd) / n;

bound = abs(norm(L*V, 'fro')) / n;


% plot
figure;
title('Convergence of Stochastic Gradient Descent');
xlabel('iteration');
ylabel('trace(X^TL^TLX)');
% fprintf('hebbian khaet time: %.4f, accuracy: %.4f\n', time.khaet, acc.khaet);
% fprintf('sgd time: %.4f, accuracy: %.4f\n', time.sgd, acc.sgd);
% fprintf('gd time: %.4f, accuracy: %.4f\n', time.gd, acc.gd);

fprintf('hebbian time: %.4f, accuracy: %.4f\n', time.hebb, acc.hebb);
fprintf('svd time: %.4f, accuracy: %.4f\n', time.svd, acc.svd);

hold on;
% plot(info_khaet.iter, info_khaet.cost, '.-', 'DisplayName', 'Hebbian khaet');


% reduce gd result
% gd_resl = length([info_gd.metric]);
% interv = floor(gd_resl / 20);
% gd_metric = fliplr([info_gd.metric]);
% gd_metric = gd_metric(1:interv:length(gd_metric));
% gd_metric = fliplr(gd_metric);
% plot(1:length(gd_metric), gd_metric, '.-', 'DisplayName', 'GD');

% sgd_metric = [info_sgd.metric];
% sgd_metric(end) = [];
% plot(1:length(sgd_metric), sgd_metric, '.-', 'DisplayName', 'R-SGD');

plot(info_hebb.iter, info_hebb.cost, '.-', 'DisplayName', 'Hebbian');
plot(info_hebb.iter, bound*ones(size(info_hebb.iter)), '--', 'DisplayName', 'SVD');

legend('Location', 'SouthEast');
hold off;

% function X = postProcessEmbedding(X, D2)
% X = D2 * X;
% 
% end

function label = cluster(U, nlabel)

U = U ./ vecnorm(U, 2);
label = litekmeans(U, nlabels, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates', 10);

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
