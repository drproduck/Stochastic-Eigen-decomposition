clear;
load('20NewsHome')
fea(1,1:100)
f = fopen('stop_word_inds');
inds = fscanf(f, '%d');
f = fopen('word_name');
word = textscan(f, '%s');
word = word{1}
sum_col = sum(fea, 1);
color = zeros(size(fea,2),1);
color(inds) = 1;

[val, rank_to_word_idx] = sort(sum_col, 'descend');
word_idx_to_rank = zeros(size(fea,2), 1);
for i = 1:size(fea, 2)
    word_idx_to_rank(rank_to_word_idx(i)) = i;
end
color = color(word_idx_to_rank);
inds = word_idx_to_rank(inds);
x = 1:size(fea,2);
figure(1)
scatter(x(1:1000), val(1:1000), 1, color(1:1000));
text(x(inds), val(inds), word, 'FontSize',8);
figure(2)
scatter(x, val, 0.5);
