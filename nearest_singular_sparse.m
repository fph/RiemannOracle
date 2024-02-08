function problem = nearest_singular_sparse(structure, A, use_hessian)
% Create a Manopt problem structure for the nearest singular sparse matrix

if isempty(structure)
    structure = double(A ~= 0);
end

n = size(A, 2);
if isreal(A)
    problem.M = spherefactory(n);
else
    problem.M = spherecomplexfactory(n);
end

% populate the struct with generic functions that include the regularization as parameters
problem.gencost  = @(epsilon, y, v, store) cost(structure, A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(structure, A, epsilon, y, v, store);
if use_hessian
    problem.genehess = @(epsilon, y, v, w, store) ehess(structure, A, epsilon, y, v, w, store);
end
problem.genminimizer = @(epsilon, y, v, store) minimizer(structure, A, epsilon, y, v, store);

% compute the value of the constraint (A+E)v + epsilon*y.
% We need this for the extended Lagrangian update
% Note that this constraint is not zero when epsilon is nonzero
problem.genconstraint = @(epsilon, y, v, store) constraint(structure, A, epsilon, y, v, store);

% populate functions from the Manopt interface with zero regularization
problem = apply_regularization(problem, 0, 0);

% additional function outside of the main interface to recover
% a more exact solution in the case of rank drops, enforcing exact
% zeros in U'*r
problem.recover_exact = @(v, tol) recover_exact(structure, A, v, tol);

end

% fill values in the 'store', a caching structure used by Manopt
% with precomputed fields that we need in other functions as well
function store = populate_store(structure, A, epsilon, y, v, store)
    if ~isfield(store, 'r')
        Av = A * v;
        store.Av = Av;
        store.r = -Av - epsilon * y;
        d = 1./ (structure * (conj(v) .* v) + epsilon);
        d(~isfinite(d)) = 0;
        store.d = d;
    end    
end

function [cf, store] = cost(structure, A, epsilon, y, v, store)
    store = populate_store(structure, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    cf = sum(conj(r) .* r .* d);
end

function [eg, store] = egrad(structure, A, epsilon, y, v, store)
    store = populate_store(structure, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    eg = -2*(A' * z + (structure' * (conj(z) .* z) .* v)); 
end

function E = minimizer(structure, A, epsilon, y, v, store)
    store = populate_store(structure, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    E = z .* (v' .* structure);
end

function [ehw, store] = ehess(structure, A, epsilon, y, v, w, store)
    store = populate_store(structure, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    rightpart = -A*w - z .* (structure * (conj(v) .* w + conj(w) .* v));
    dz = store.d .* rightpart;
    d1 = (structure' * (conj(z) .* z)) .* w;
    d2 = (structure' * ((conj(z) .* dz) + (conj(dz) .* z))) .* v;
    ehw = -2 * (A'*dz + d1 + d2);
end

function [prod, store] = constraint(structure, A, epsilon, y, v, store)
    store = populate_store(structure, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    E = z .* (v' .* structure);
    prod = store.Av + E * v;
end

function v_reg = recover_exact(structure, A, v, tol)
    store = struct();
    store = populate_store(structure, A, 0, 0, v, store);
    r = store.r;
    d = store.d;
    r_reg = r;
    r_reg(abs(d) > tol) = 0;
    v_reg = - A \ r_reg;
end