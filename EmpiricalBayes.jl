
# Returns (1/2) log p(X^T) given S, m, s and alpha 
function likelog(G_vec, G_matrix; s = 1, alpha = 1.5)
    N = length(G_vec)
    Lambda = Diagonal([s^2 * k^(-2.0 * alpha - 1.0) for k in 1:N]) 
    B =  inv(Lambda) + G_matrix
    V = inv(B)
    return dot(G_vec, V * G_vec) - logdet(Lambda) - logdet(B)
end
    
# Calculates the empirical bayes estimator based on sample path and specified basis.
function empBayes(path, basis)
    G_vec = giraVector(path, basis)
    G_matrix = giraMatrix(path, basis)
    return Optim.minimizer(optimize(x -> -likelog(G_vec, G_matrix, s = x[1], alpha = x[2]), [1.,1.])  )
end

