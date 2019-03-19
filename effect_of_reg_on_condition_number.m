load('circledata');

for pad = 0:10
	L = getRBFLaplacian(fea);
	n = size(L,1);
	[u,s] = eig(L+pad*(speye(n)));
	s = diag(s);
	max(s) / min(s)
	[~,p] = sort(s,'descend');
	u = u(:,p(1:2));
	figure;
	scatter(u(:,1),u(:,2),[],gnd);

end
