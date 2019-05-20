function [U, S] = nystrom(W, k, n_sample)

% nystrom spectral clustering


n = size(W,1);
samples = randsample(n, n_sample);
C = W(:, samples)
A = C(samples, :)
nonsamples = setdiff(1:n, samples)
B = W(samples, nonsamples)

Asi = sqrtm(pinv(A));
Q = A + Asi * B * B' * Asi;

[U, S] = eigs(Q, k);
U = [A; B'] * Asi * U * diag(diag(S) .^ -0.5);

% S = (n / n_sample) .* S;
% U = [U; B'*U* diag(1 ./ diag(S))];
% U = sqrt(n_sample / n) .* C * U * diag(1 ./ diag(S));

end
