function N = vecnorm(A, axis)

% return norms along the dimension of A

N = sqrt(sum(A.^2, axis));
