function r = makesphere(n, R, tsl, noi)
r = randn(n, 3); % Use a large n
r = r ./ sqrt(sum(r.^2,2)) .* R;
noise = rand(size(r)) .* noi
r = r + noise + repmat(tsl, n, 1)

end
   
