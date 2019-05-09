function [colors, onehot_labels] = getColors(labels)

n_colors = length(unique(labels));
distinct_colors = distinguishable_colors(n_colors);
onehot_labels = zeros(length(labels), n_colors);
n = length(labels);

% labels shoud be column-wise
if min(labels) == 0
	onehot_labels(labels*n + (1:n)') = 1;
else
	onehot_labels(labels*n + (1:n)') = 1;
end

colors = onehot_labels * distinct_colors;
