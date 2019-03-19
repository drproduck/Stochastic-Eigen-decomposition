function L = getRBFLaplacian(fea, opt)

defaults.s = -1;
defaults.reg = -1;
defaults.normalize = 1;

if ~exist('opt','var') || isempty(opt)
	opt = struct()

end

opt = mergeOptions(defaults, opt);

s = opt.s;
reg = opt.reg;
normalize = opt.normalize;

n = size(fea, 1);

d = pdist(fea, 'squaredeuclidean');
A = squareform(d);

if s <= 0;
	s = getSigma(fea);

end

W = exp(-A ./ (2.0*s^2));

d = sum(W, 2);
d = max(d, 1e-15);

if reg < 0
	reg = sum(d) / n;

end

d = d + reg;

if normalize
	D = sparse(1:n,1:n,d.^(-0.5));
	L = D*W*D;

else
	D = sparse(1:n,1:n,d);
	L = D - W;

end

L = (L + L') ./ 2;
	
end
