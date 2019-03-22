function s = getSigma(fea)

if (size(fea,1) > 4000)
	rperm = randperm(size(fea,1));
	feasample = fea(rperm(1:4000),:);
else
	feasample = fea;

end

% euclidean
D = pdist(feasample);

s = median(D);

end
