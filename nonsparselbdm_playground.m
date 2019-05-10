% load('circledata.mat');
clear;
load('Orig.mat')
% rng(9999);

%[L, idx, D1, D2] = getSparseBipartite(fea, 1000, 3, 'uniform');
[L, idx] = getBipartite(fea, 200);

[n,d] = size(L);
k = 11;
nlabel = 10;
npass = 10;

% shared starting point
X_0 = randn(d,k);
[X_0,~] = qr(X_0, 0);



clear opt;
opt.npass = 1;
opt.batchsize = 1000;
opt.stepsize_init = 1e2; % previous 1e2
opt.stepsize_type = 'decay';
opt.checkperiod = 10;
tic;
[X_hebb, info_hebb] = stochastic_hebbian(L, k, X_0, opt);
time.hebb = toc;
% scatter2d(X_hebb, gnd(idx))
label = landmark_cluster(X_hebb, nlabel, idx);
label = bestMap(gnd, label);
acc.hebb = sum(label == gnd) / n;


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

fprintf('hebbian time: %.4f, accuracy: %.4f\n', time.hebb, acc.hebb);
fprintf('svd time: %.4f, accuracy: %.4f\n', time.svd, acc.svd);

hold on;

plot(info_hebb.iter, info_hebb.cost, '.-', 'DisplayName', 'Hebbian');
plot(info_hebb.iter, bound*ones(size(info_hebb.iter)), '--', 'DisplayName', 'SVD');

legend('Location', 'SouthEast');
hold off;


function label = cluster(U, nlabel)

U = U ./ vecnorm(U, 2);
label = litekmeans(U, nlabels, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates', 10);

end


function label = landmark_cluster(V, nlabel, idx) 
[n,~] = size(idx); V(:,1) = []; 
V = V ./ sqrt(sum(V .^2, 2)); 
reps_labels = litekmeans(V, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates', 10);
label = zeros(n, 1);
for i = 1:n label(i) = reps_labels(idx(i));
end 
end
