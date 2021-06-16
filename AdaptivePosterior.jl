# Runs MCMC based on proposal q(i +- 1 | i) = 1/4, q(i|i) = 1/2. C refers to log(1-p). Returns list of posterior J, s and z observations.
function MCMC(path, basis_fnc, iter; j0, z0, s_sq0, alpha, A, B, C)  
    x_vals = path.samplevalues; dt = path.timeinterval[2] - path.timeinterval[1];
    N = j0 
    init_basis = [basis_fnc(k) for k in 1:N]
    W = sig + (1/s_sq0) * Lambda_inv    
    # Initialization    
    j_chain = [j0]; z_chain = [z0]; s_sq_chain = [s_sq0] 
    j = j0;z = z0;s_sq = s_sq0
    for i in 1:iter
        # Gibbs sampler: s^2
        s_sq = 1 / rand(Gamma(A + (1/2)j, (B + (1/2) transpose(z) * Lambda_inv[1:j, 1:j] * z )^(-1) ))
        push!(s_sq_chain, s_sq)
        # MH step: J
        j_it = sample([j - 1, j, j + 1], Weights([0.25, 0.5, 0.25]), 1)[1]
        # Expand current basis by 10 when j increases
        if j_it > j
            new_N = N + 10; new_mu = zeros(new_N);  new_mu[1:N] = mu;                     
            new_sig = zeros(new_N, new_N);new_sig[1:N, 1:N] = sig
            for i=1:10,j=1:10
                new_mu[N+i] = vectorElement(x_vals, basis_fnc(N+i))
                new_sig[N + i, N + j] = matrixElement(x_vals, basis_fnc(N+i), basis_fnc(N+j), dt)
            end
            mu = new_mu; sig = new_sig;
            Lambda_inv = Diagonal([k^(2.0 * alpha + 1.0) for k in 1:new_N])  
            N = new_N
        end    
        W = sig + (1/s_sq) * Lambda_inv
        inv_W_it = inv(W[1:j_it, 1:j_it]); inv_W = inv(W[1:j, 1:j]);   
        mu_it = mu[1:j_it]
        # Gibbs sampler: z_1,...,z_J
        z_it = rand(MvNormal(inv_W_it * mu_it, Symmetric(inv_W_it)))
        rest = sign(j_it - j) * (2alpha + 1) * log(max(j, j_it)) + (j - j_it) * log(s_sq)
        logB_it =(1/2) * (transpose(mu_it) * inv_W_it * mu_it - logdet(W[1:j_it, 1:j_it]) - 
            (transpose(mu[1:j])* inv_W * mu[1:j] - logdet(W[1:j, 1:j])) + rest)    
        logR_it = C * (j_it - j) 
        accept_p = exp(logB_it + logR_it)
        # Acceptance / Rejection step
        if rand(Uniform()) < accept_p j = j_it; z = z_it; end
        push!(j_chain, j); push!(z_chain, z);
    end
    return (j_chain, z_chain, s_sq_chain)        
end
    
# Calculates posterior realizations from MCMC z-observations on the grid x given a basis
function func_from_coeffs(z_array, basis, x)
    M = maximum(length.(z_array))    
    basis_matrix = [basis(k).(x0) for x0 in x, k in 1:M]     
    res = zeros(length(z_array), length(x))    
    for i in 1:length(z_array)
        z = z_array[i]
        res[i,:] = basis_matrix[:, 1:length(z)] * z
    end
    return res
end   

# Returns (1/2) log p(X^T) given s and alpha
function likelog(G_vec, G_matrix; s = 1, alpha = 1.5)
    N = length(G_vec)
    Lambda = Diagonal([s^2 * k^(-2.0 * alpha - 1.0) for k in 1:N]) 
    B =  inv(Lambda) + G_matrix
    V = inv(B)
    return dot(G_vec, V * G_vec) - logdet(Lambda) - logdet(B)
    
end
    
# Calculates the empirical bayes estimator
function empBayes(path, basis)
    G_vec = giraVector(path, basis)
    G_matrix = giraMatrix(path, basis)
    return Optim.minimizer(optimize(x -> -likelog(G_vec, G_matrix, s = x[1], alpha = x[2]), [1.,1.])  )
end
