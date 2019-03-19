function [x,cost,truecost] = stochastic_rg(A, k);

	n = size(A,1);
	stepsize = 1e-3;
	niters = 100;

	cost = zeros(niters,1);


	% initialize
	x = randn(n,k);
	x = x ./ sqrt(sum(x.^2,1));

	for i = 1:niters
		% gradient
		g = -(1/n)*(A*x);
		g = (eye(n) - x*x')*g;

		% update
		x = x + stepsize .* g;
		[x,~] = qr(x,0);

		cost(i) = (1/2)*(1/n)*trace(x'*A*x);

	end

	[u,s] = eigs(A,k);
	truecost = (1/2)*(1/n)*trace(u'*A*u);
	

end
