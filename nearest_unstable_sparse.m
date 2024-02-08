function problem = nearest_unstable_sparse(structure, target, A)
% Create a Manopt problem structure for the nearest omega-stable sparse matrix
%
% problem = nearest_stable_sparse(structure, target, A, use_hessian)
%
% see nearest_singular_sparse for more comments on this interface

if isempty(structure)
    structure = A ~= 0;
end

if strcmp(target, 'Schur')
    target = @(x) outside_disc(x, 1);
elseif strcmp(target, 'Hurwitz')
    target = @(x) right_of(x, 0);
end

n = size(A, 2);
problem.M = spherecomplexfactory(n);


problem.gencost  = @(epsilon, y, v, store) cost(structure, target, A, epsilon, y, v, store);
problem.genegrad = @(epsilon, y, v, store) egrad(structure, target, A, epsilon, y, v, store);
problem.genminimizer = @(epsilon, y, v, store) minimizer(structure, target, A, epsilon, y, v, store);
problem.genconstraint = @(epsilon, y, v, store) constraint(structure, target, A, epsilon, y, v, store);
problem.recover_exact = @(v, tol) recover_exact(structure, target, A, v, tol);

problem = apply_regularization(problem, 0, 0);
end

function store = populate_store(structure, target, A, epsilon, y, v, store)
    if ~isfield(store, 'r')
        Av = A * v;
        r0 = -Av - epsilon * y;
        d = 1./ (structure * (conj(v) .* v) + epsilon);
        d(~isfinite(d)) = 0;
        
        Dv = d .* v;
        lambda = target(-(Dv' * r0) / (Dv'*v));
        store.Av = Av;
        store.d = d;
        store.lambda = lambda;
        store.r = lambda * v + r0;
    end
end

function [cf, store] = cost(structure, target, A, epsilon, y, v, store)
    store = populate_store(structure, target, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    cf = sum(conj(r) .* r .* d);
end

function [eg, store] = egrad(structure, target, A, epsilon, y, v, store)
    store = populate_store(structure, target, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    lambda = store.lambda;
    z = d .* r;
    eg = -2*(A' * z - conj(lambda)*z + (structure' * (conj(z) .* z) .* v)); 
end

function [E, lambda, store] = minimizer(structure, target, A, epsilon, y, v, store)
    store = populate_store(structure, target, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    E = z .* (v' .* structure);
    lambda = store.lambda;
end

% function [eh, store] = ehess(structure, A, epsilon, y, v, w, store)
%     store = populate_store(structure, A, epsilon, y, v, store);
%     r = store.r;
%     d = store.d;
%     z = d .* r;
%     rightpart = -A*w - z .* (structure * (conj(v) .* w + conj(w) .* v));
%     dz = store.d .* rightpart;
%     d1 = (structure' * (conj(z) .* z)) .* w;
%     d2 = (structure' * ((conj(z) .* dz) + (conj(dz) .* z))) .* v;
%     eh = -2 * (A'*dz + d1 + d2);
% end
% 

function [prod, store] = constraint(structure, target, A, epsilon, y, v, store)
    [E, lambda, store] = minimizer(structure, target, A, epsilon, y, v, store);
    prod = -store.r + E * v;
end

function v_reg = recover_exact(structure, A, v, tol)
    TODO - unfinished
    store = struct();
    store = populate_store(structure, A, v, 0, 0, store);
    r = store.r;
    d = store.d;
    r_reg = r;
    r_reg(abs(d) > tol) = 0;
    v_reg = - A \ r_reg;
end