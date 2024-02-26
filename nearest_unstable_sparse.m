function problem = nearest_unstable_sparse(structure, target, A)
% Create a Manopt problem structure for the nearest omega-stable sparse matrix
%
% problem = nearest_stable_sparse(structure, target, A, use_hessian)
%
% see nearest_singular_sparse for more comments on this interface

if isempty(structure)
    structure = double(A ~= 0);
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

problem.gencost_with_prescribed_lambda = @(epsilon, y, v, lambda, store) gencost_with_prescribed_lambda(structure, target, A, epsilon, y, v, lambda, store);

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
        store.normAv = norm(Av);        
    end
end

function [cf, store] = cost(structure, target, A, epsilon, y, v, store)
    store = populate_store(structure, target, A, epsilon, y, v, store);
    r = store.r;
    d = store.d;
    cf = sum(conj(r) .* r .* d);
end

function [cf, store] = gencost_with_prescribed_lambda(structure, target, A, epsilon, y, v, lambda, store)
    store = populate_store(structure, target, A, epsilon, y, v, store);
    r = lambda * v - store.Av - epsilon * y;
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

% The Hessian for this problem is messier, because to compute it we need
% the Hessian of the squared distance function, and this depends on target.
%
% For instance, if target = 'Hurwitz' the Hessian of the squared distance 
% function is 0, and if target = 'Schur' it is 2I.
%
% For now we just avoid the problem, we can live with first-order methods.
%
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
    prod = store.Av + E * v - v * lambda;
end

function [v_reg cost_reg] = recover_exact(structure, target, A, v, tol)
    store = populate_store(structure, target, A, 0, 0, v, struct());
    r = store.r;
    d = store.d;
    % the lambda from a model with a small perturbation might be more accurate
    % because we avoid the exact singularity, and dlambda/dv = 0.
    % store2 = populate_store(structure, target, A, 1e-12, 0, v, struct());
    % lambda = store2.lambda;
    lambda = store.lambda;
    r_reg = r;
    r_reg(abs(d) > tol) = 0;
    v_reg = - (A - lambda*speye(length(A))) \ r_reg;
    store_reg = populate_store(structure, target, A, 0, 0, v_reg, struct());
    cost_reg = sum(conj(r_reg) .* r_reg .* store_reg.d);
end