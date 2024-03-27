% unit tests for solve_system_svd
import matlab.unittest.TestCase
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.RelativeTolerance

testCase = TestCase.forInteractiveUse;

% tall-thin case

rng(0);
m = 10;
p = 3;
epsilon = 1;
M = randn(m, p);
[U1, S, ~] = svd(M, 'econ');
d = 1 ./ (diag(S).^2 + epsilon);
r1 = randn(m, 1);
r2 = randn(size(U1,2), 1);
[x, U1tx, cf] = solve_system_svd(U1, d, epsilon, r1, r2);

testCase.verifyThat((M*M'+epsilon*eye(m))*x, IsEqualTo(r1 + U1*r2, ...
    "Within", RelativeTolerance(sqrt(eps))));

testCase.verifyThat(U1'*x, IsEqualTo(U1tx, ...
    "Within", RelativeTolerance(sqrt(eps))));

cftrue = (r1 + U1*r2)' / (M*M'+epsilon*eye(m)) * (r1 + U1*r2);

testCase.verifyThat(cf, IsEqualTo(cftrue, ...
    "Within", RelativeTolerance(sqrt(eps))));

% short-fat case

rng(0);
m = 10;
p = 15;
epsilon = 1;
M = randn(m, p);
[U1, S, V] = svd(M, 'econ');
d = 1 ./ (diag(S).^2 + epsilon);
r1 = randn(m, 1);
r2 = randn(size(U1,2), 1);
[x, U1tx, cf] = solve_system_svd(U1, d, epsilon, r1, r2);

testCase.verifyThat((M*M'+epsilon*eye(m))*x, IsEqualTo(r1 + U1*r2, ...
    "Within", RelativeTolerance(sqrt(eps))));

testCase.verifyThat(U1'*x, IsEqualTo(U1tx, ...
    "Within", RelativeTolerance(sqrt(eps))));

cftrue = (r1 + U1*r2)' / (M*M'+epsilon*eye(m)) * (r1 + U1*r2);

testCase.verifyThat(cf, IsEqualTo(cftrue, ...
    "Within", RelativeTolerance(sqrt(eps))));
