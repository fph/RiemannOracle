function problem = nearest_singular(structure, A, epsilon, y, use_hessian)

if isempty(structure)
    structure = A ~= 0;
end
if not(exist('epsilon', 'var')) || isempty(epsilon)
    epsilon = 0.0;
end
if not(exist('y', 'var')) || isempty(y)
    y = 0;
end

n = size(A, 2);
problem.M = spherecomplexfactory(n);

problem.cost  = @(v, store) cost(structure, A, v, epsilon, y, store);
problem.egrad = @(v, store) egrad(structure, A, v, epsilon, y, store);
if use_hessian
    problem.ehess = @(v, w, store) ehess(structure, A, v, epsilon, y, w, store);
end
problem.minimizer = @(v, store) minimizer(structure, A, v, epsilon, y, store);
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
    z = d .* r;
    eg = -2*(A' * z + (structure' * (conj(z) .* z) .* v)); 
end

function E = minimizer(structure, A, v, epsilon, y, store)
    store = populate_store(structure, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    E = z .* (v' .* structure);
end

function [eh, store] = ehess(structure, A, v, epsilon, y, w, store)
    store = populate_store(structure, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    rightpart = -A*w - z .* (structure * (conj(v) .* w + conj(w) .* v));
    dz = store.d .* rightpart;
    dEtz = (structure' * (conj(z) .* z)) .* w + (structure' * (conj(dz) .* z)) .* v;
    Etdz = (structure' * (conj(z) .* dz)) .* v;
    eh = -2 * (A'*dz + Etdz + dEtz);
end
