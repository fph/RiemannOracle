function problem = nearest_stable_sparse(structure, target, A)
% Create Manopt problem structure for the nearest omega-stable sparse matrix
%
% problem = nearest_stable_sparse(structure, target, A, use_hessian)

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


problem.gencost  = @(v, epsilon, y, store) cost(structure, target, A, v, epsilon, y, store);
problem.genegrad = @(v, epsilon, y, store) egrad(structure, target, A, v, epsilon, y, store);
problem.genminimizer = @(v, epsilon, y, store) minimizer(structure, target, A, v, epsilon, y, store);
problem.genconstraint = @(v, epsilon, y, store) constraint(structure, target, A, v, epsilon, y, store);
% problem.recover_exact = @(v, tol) recover_exact(structure, target, A, v, tol);

problem = apply_regularization(problem, 0, 0, false);
end

function store = populate_store(structure, target, A, v, epsilon, y, store)
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

function [cf, store] = cost(structure, target, A, v, epsilon, y, store)
    store = populate_store(structure, target, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    cf = sum(conj(r) .* r .* d);
end

function [eg, store] = egrad(structure, target, A, v, epsilon, y, store)
    store = populate_store(structure, target, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    lambda = store.lambda;
    z = d .* r;
    eg = -2*(A' * z - conj(lambda)*z + (structure' * (conj(z) .* z) .* v)); 
end

function [E, lambda] = minimizer(structure, target, A, v, epsilon, y, store)
    store = populate_store(structure, target, A, v, epsilon, y, store);
    r = store.r;
    d = store.d;
    z = d .* r;
    E = z .* (v' .* structure);
    lambda = store.lambda;
end

% function [eh, store] = ehess(structure, A, v, epsilon, y, w, store)
%     store = populate_store(structure, A, v, epsilon, y, store);
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
% % compute the value of the constraint (A+E)v --- note that this is not zero
% % if epsilon is nonzero.
% function [prod, store] = constraint(structure, A, v, epsilon, y, store)
%     store = populate_store(structure, A, v, epsilon, y, store);
%     r = store.r;
%     d = store.d;
%     z = d .* r;
%     E = z .* (v' .* structure);
%     prod = store.Av + E * v;
% end

% Tries to recover an "exact" v from a problem converging to a rank-drop
% point
% function v_reg = recover_exact(structure, A, v, tol)
%     store = struct();
%     store = populate_store(structure, A, v, 0, 0, store);
%     r = store.r;
%     d = store.d;
%     r_reg = r;
%     r_reg(abs(d) > tol) = 0;
%     v_reg = - A \ r_reg;
% end