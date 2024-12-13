function [z, delta, cf] = solve_system_svd(U1, VS, d, epsilon, r1, r2)
% solve a linear system (M*M'+epsilon*I) z = r
%
% Input: U1, V*S from the (possibly thin) SVD of M, 
% and the precomputed d = 1 ./ (s^2+epsilon).
% r1, r2 such that r = (r1 + U1*r2)
%
% Return z, M'*z, and also r'*(MM'+epsilon*I)^{-1}r, which 
% are cheap to compute and are useful, too, sometimes.

% relies on the expansion z = U1*diag(D)*U1'*r + U2*1/epsilon*I*U2'*r
% after replacing U2*U2' = I - U1*U1'.
% Also note that U2*U2' = 0 unless M is tall-thin

U1tr1 = U1'*r1;
U1tr = r2 + U1tr1;
U1tz = d .* U1tr;

if size(U1,1) > size(U1,2) % U tall thin
    orthogonal_part = r1 - U1*U1tr1;
else
    orthogonal_part = 0;
end
if norm(orthogonal_part) == 0 && epsilon == 0
    epsilon = 1; % avoid NaNs
end
z = U1*U1tz + orthogonal_part/epsilon;
if nargout > 1
    delta = VS * U1tz;
end
if nargout > 2
    % U1tr'*U1tz, but working also for multiple right-hand sides as the
    % trace product
    cf = real(sum(sum(conj(U1tr).*U1tz))) + norm(orthogonal_part, 'fro')^2/epsilon;
end
