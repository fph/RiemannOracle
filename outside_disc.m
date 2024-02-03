function x = outside_disc(x, r)
% project x outside of the disc |z| <= r

if not(exist('r', 'var'))
    r = 1;
end

if abs(x) < r
    x = x/abs(x)*r;
end
