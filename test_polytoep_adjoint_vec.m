% unit tests for polytoep_adjoint_vector
import matlab.unittest.TestCase
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.RelativeTolerance

testCase = TestCase.forInteractiveUse;

rng(0);
m = 3;
n = 2;
A0 = zeros(m,n); A1 = zeros(m,n); A2 = zeros(m,n);
A = [];
A(:,:,1) = A0; A(:,:,2) = A1; A(:,:,3) = A2;

w1 = zeros(m,1); w2 = zeros(m,1); w3 = zeros(m,1); w4 = zeros(m,1); w5 = zeros(m,1);
w = [w1; w2; w3; w4; w5];

v = [A0'*w1+A1'*w2+A2'*w3;
     A0'*w2+A1'*w3+A2'*w4;
     A0'*w3+A1'*w4+A2'*w5];

testCase.verifyThat(polytoep_adjoint_vec(A, 2, w), IsEqualTo(v, ...
    "Within", RelativeTolerance(sqrt(eps))));

rng(0);
m = 5;
n = 4;
k = 3;
d = 6;
A = randn(m, n, k+1) + 1i * randn(m, n, k+1);
w = randn(m, k+d+1) + 1i * randn(m, k+d+1);

testCase.verifyThat(polytoep_adjoint_vec(A, d, w), IsEqualTo(polytoep(A, d)' * w(:), ...
    "Within", RelativeTolerance(sqrt(eps))));
