# Calculates S_i
function matrixElement(x_vals, b1, b2, dt)
	return sum(b1.(x_vals) .* b2.(x_vals) * dt)
end

# Calculates S 
function giraMatrix(path, basis)
	N =  length(basis)
	Sig = Array{Float64}(undef, N, N) 
	for i=1:N, j=1:N
		Sig[i,j] = matrixElement(path.samplevalues, basis[i], basis[j], path.timeinterval[2] - path.timeinterval[1])
	end
	return Symmetric(Sig) 
end

# Calculates m_i
function vectorElement(x_vals, b)
	return sum(b.(x_vals[1:(length(x_vals)-1)]) .* diff(x_vals)))
    end    
    
# Calculates m 
function giraVector(path, basis)
	x_vals = path.samplevalues	
	N =  length(basis)
	return [vectorElement(x_vals, basis[i]) for i in 1:N]
end    

# Calculates A_n with grid and basisfunctions as input
function phi_matrix(x, basis)
	return [basis[j](x[i]) for i in 1:length(x), j in 1:length(basis)]
end
# Specifies prior in terms of s, alpha and a basis
function prior_dist(s, alpha, basis)
    N = length(basis)
    d = GaussianVector(sparse(Diagonal([s * k^(-alpha -0.5) for k in 1:N])))
    return GaussianProcess(basis, d)
end  
# Calculates posterior based on observed data from a model 
function post_from_data(mod, path, basis; alpha = 1.5, s = 1.0)
    N = length(basis)
    prior = prior_dist(s, alpha, basis)
    return calculateposterior(prior, path, mod)
end  
# Extracts (theta(t_1),...,theta(t_n)) | X^T parameters from posterior based on grid and basis
function post_pars(post, x, basis)
    sigma_hat = post.distribution.var
    mu_hat = post.distribution.mean;
    phi = phi_matrix(x, basis)
    post_mean = phi * mu_hat
    post_var = phi * sigma_hat * transpose(phi)
    return (post_mean, post_var)
end  
