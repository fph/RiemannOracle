function x = right_of(x, r)
% project x to the right of the line real(z) = r

if not(exist('r', 'var'))
    r = 0;
end

if real(x) < r
    x = r + 1i * imag(x);
end
