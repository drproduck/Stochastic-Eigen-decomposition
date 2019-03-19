figure(1);
x = [1,2,3];
y = [7.3233 3.6429 2.5759 ];
bar(x,y);
xticklabels({'Hebbian', 'R-SGD', 'SVD'})
title('run-time (s)');
figure(2);
x = [1,2,3];
y = [47.92 54.32 48.93];
bar(x,y);
xticklabels({'Hebbian', 'R-SGD', 'SVD'})
title('clustering accuracy (%)');

