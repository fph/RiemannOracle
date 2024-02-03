function problem = nearest_singular(structure, A, epsilon, y)

if isempty(structure)
    structure = A ~= 0;
end
if not(exist('epsilon', 'var'))
    epsilon = 0.0;
end
if not(exist('y', 'var'))
    y = 0;
end

n = size(A, 2);
manifold = spherecomplexfactory(n);
problem.M = manifold;

% Define the problem cost function and its Euclidean gradient.
problem.cost  = @(v, store) cost(structure, A, v, epsilon, y, store);
problem.egrad = @(v, store) egrad(structure, A, v, epsilon, y, store);
end

function store = populate_store(structure, A, v, epsilon, y, store)
    if ~isfield(store, 'r')
        store.r = -A*v - epsilon * y;
        d = 1./ (structure * (conj(v) .* v) + epsilon);
        d(~isfinite(d)) = 0;
        store.d = d;
    end    
end

function [cf, store] = cost(structure, A, v, epsilon, y, store)
    store = populate_store(structure, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    cf = sum(conj(r) .* r .* d);
end
function [eg, store] = egrad(structure, A, v, epsilon, y, store)
    store = populate_store(structure, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    z = -r .* d;
    eg = 2*(A' * z - (structure' * (conj(z) .* z) .* v)); 
end
