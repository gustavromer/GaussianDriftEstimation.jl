using Roots, Random, Distributions,BayesianNonparametricStatistics,LinearAlgebra, SparseArrays , Plots, Optim, LaTeXStrings, ProgressMeter, StatsBase


function unit_fourier(k)
  sqrt_two = sqrt(2.0) 
  if k == 0
    return x -> 1.0
  elseif isodd(k)
    return x -> sqrt_two*sinpi(float(k+1)*x)
  else
    return x -> sqrt_two*cospi(float(k)*x)
  end
end

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


function MCMC(path, basis_fnc, iter; j0, z0, s_sq0, alpha, A, B, C) 
    
    x_vals = path.samplevalues
    dt = path.timeinterval[2] - path.timeinterval[1]
    
    
    N = j0
    
    init_basis = [basis_fnc(k) for k in 1:N]
    #init_basis = [basis_fnc(k) for k in 0:(N-1)]
    
    Lambda_inv = Diagonal([k^(2.0 * alpha + 1.0) for k in 1:N])    
    mu = giraVector(path, init_basis)
    sig = giraMatrix(path, init_basis)
    
    
    W = sig + (1/s_sq0) * Lambda_inv
    
    
    j_chain = [j0]
    z_chain = [z0]
    s_sq_chain = [s_sq0]
    
    j = j0
    z = z0
    s_sq = s_sq0

    
    @showprogress for i in 1:iter
        # works when changed to ordinary gamma. change scale
        s_sq = 1 / rand(Gamma(A + (1/2)j, (B + (1/2)z' * Lambda_inv[1:j, 1:j] * z )^(-1) ))
        push!(s_sq_chain, s_sq)
        
        # expand current basis if j increases
        j_it = sample([j - 1, j, j + 1], Weights([0.25, 0.5, 0.25]), 1)[1]
        if j_it > j
            new_N = N + 10
            
            
            new_mu = zeros(new_N)
            new_mu[1:N] = mu
            
            new_sig = zeros(new_N, new_N)
            new_sig[1:N, 1:N] = sig
            
            for i in 1:10
                new_mu[N+i] = vectorElement(x_vals, basis_fnc(N+i))
                for j in 1:10
                    new_sig[N + i, N + j] = matrixElement(x_vals, basis_fnc(N+i), basis_fnc(N+j), dt)
                end
            end
            
            mu = new_mu
            sig = new_sig
            Lambda_inv = Diagonal([k^(2.0 * alpha + 1.0) for k in 1:new_N])  
            
            N = new_N
        end    
        
        W = sig + (1/s_sq) * Lambda_inv
        
        inv_W_it = inv(W[1:j_it, 1:j_it])
        inv_W = inv(W[1:j, 1:j])
        
        mu_it = mu[1:j_it]
        
        z_it = rand(MvNormal(inv_W_it * mu_it, Symmetric(inv_W_it)))
                
        rest = sign(j_it - j) * (2alpha + 1) * log(max(j, j_it)) + (j - j_it) * log(s_sq)
        logB_it =(1/2) * (mu_it' * inv_W_it * mu_it - logdet(W[1:j_it, 1:j_it]) - 
            (mu[1:j]' * inv_W * mu[1:j] - logdet(W[1:j, 1:j])) + rest)
            
        logR_it = C * (j_it - j)
    
        accept_p = exp(logB_it + logR_it)
            
        if rand(Uniform()) < accept_p
            j = j_it     
            z = z_it
        end
        
        push!(j_chain, j)
        push!(z_chain, z)
    end
            
    return (j_chain, z_chain, s_sq_chain)        
end



# calculate frequentist coverage

function emp_check_cov(theta, x, mod, sde, basis;  N = 10^3)
    
    obs_theta = theta.(x)
    
    sim_cov = zeros(N)
    point_cov = zeros(N, length(x))
    
    prog = Progress(N)
    Threads.@threads for i in 1:N
        path = rand(sde)
        ests = empBayes(path, basis)
        post = post_from_data(mod, path, basis, alpha = ests[2], s = ests[1])
        pars = post_pars(post, x, basis)
        sim_margin = simul_band(pars[2])
        point_margin = point_band(pars[2])
        dist = abs.(obs_theta - pars[1]) 
        push!(sim_cov, all(dist .< sim_margin))
        point_cov[i,:] = dist .< point_margin
        
        # update progress
        next!(prog)
    end
    
    return (mean(point_cov, dims = 1), mean(sim_cov))
end


function check_cov(theta, s, alpha, x, mod, sde, basis;  N = 10^3)
    
    obs_theta = theta.(x)
    
    sim_cov = zeros(N)
    point_cov = zeros(N, length(x))
    
    prog = Progress(N)
    Threads.@threads for i in 1:N
        path = rand(sde)
        post = post_from_data(mod, path, basis, alpha = alpha, s = s)
        pars = post_pars(post, x, basis)
        sim_margin = simul_band(pars[2])
        point_margin = point_band(pars[2])
        dist = abs.(obs_theta - pars[1]) 
        push!(sim_cov, all(dist .< sim_margin))
        point_cov[i,:] = dist .< point_margin
        
        # update progress
        next!(prog)
    end
    
    return (mean(point_cov, dims = 1), mean(sim_cov))
end
    
function emp_bayes_band(mu, basis, x, mod; p = 0.95, N = 10^3,  marg = false)
    
    # We are reusing the path
    
    samps = zeros(N, length(x))
    
    sde = SDE(mu, mod)
    
    prog = Progress(N)
    Threads.@threads for i in 1:N
        path = rand(sde)
        
        bayes_est = empBayes(path, basis)
        
        post = post_from_data(mod, 
            path, 
            basis, 
            alpha = bayes_est[2],  
            s = bayes_est[1])

        
        samps[i, :] = rand(post).(x) - mean(post).(x)
        
         # update progress
        next!(prog)
    end
    
    return sim_simul_band(samps, zeros(length(x)), p = p, marg = marg)
end

function fixed_band(post, x; p = 0.95, N = 10^3, marg = false)
    samps = [rand(post).(x0) for i in 1:N, x0 in x]
    mu_vec = mean(post).(x)
    return sim_simul_band(samps,mu_vec, p = p, marg = marg)
end
    
    
function sim_simul_band(samps, mu_vec; p = 0.95, marg = false)
    N = size(samps, 1)
    dists = [maximum(abs.(samps[i, :] - mu_vec)) for i in 1:N]
    ranks = sortperm(dists)[1:ceil(Int, p * N)]
    used_samps = samps[ranks,:]
    res = hcat(vec(minimum(used_samps, dims =1)), vec(maximum(used_samps, dims =1)))
        
    if marg    
    # Add marginal bands
    a = 1 - p
    obs = size(samps, 2)
    res = (Transpose(hcat([quantile(samps[: , j], [a/2, 1 - a/2]) for j in 1:obs]...)), res)
            
    end
        
    return res    
end    

function simul_band(sig; p = 0.95, scale = true, N = 10^5, step = 0.01, max = 50)
    sig = Symmetric(sig + 0.000001I)
    d = MvNormal(sig)
    sample = rand(d,N)
    
    if scale
        rect = sqrt.(diag(sig)) / sqrt(sig[1,1])
    else
        rect = ones(size(sig,1))
    end
    
    f(R) = mean(all(abs.(sample) .<= R * rect, dims = 1)) - p
    
    R = find_zero(f, (0, max))
    
    return R * rect
    
end


function point_band(sig; p = 0.95)
   a = 1-p
   quantile(Normal(), 1 - a/2) * sqrt.(diag(sig))
end

function prior_dist(s, alpha, basis)
    N = length(basis)
    d = GaussianVector(sparse(Diagonal([s * k^(-alpha -0.5) for k in 1:N])))
    
    return GaussianProcess(basis, d)
end
    
function post_from_data(mod, path, basis; alpha = 0.7, s = 1.0)
    N = length(basis)
   
    #d = GaussianVector(sparse(Diagonal([s * k^(-alpha -0.5) for k in 1:N])))
    #prior = GaussianProcess(basis, d)
    
    prior = prior_dist(s, alpha, basis)
    
    return calculateposterior(prior, path, mod)
end

function rad_fourier(k)
  if k == 0
    return x -> 1/ sqrt(2pi)
  elseif isodd(k)
    return x -> sin(float(k+1)*x / 2.) / sqrt(pi)
  else
    return x -> cos(float(k)*x / 2.) / sqrt(pi)
  end
end

function phi_matrix(x, basis)
    res = zeros(length(x), length(basis));
    for i in 1:length(x)
       for j in 1:length(basis)
            res[i,j] = basis[j](x[i])
        end
    end
    
    return res
end


function post_pars(post, x, basis)
    sigma_hat = post.distribution.var
    mu_hat = post.distribution.mean;
    
    phi = phi_matrix(x, basis)

    post_mean = phi * mu_hat
    post_var = phi * sigma_hat * transpose(phi)
    return (post_mean, post_var)
end  


function post_plot(post_mean, post_var; a = 0.05, scale = true)
    
# calculating pointwise credibility band
    pointwise_CB = point_band(post_var, p = 1 -a)
    # calculating sim credibility band
    joint_rect = simul_band(post_var, scale = scale, p = 1 - a)
    
    p = plot()
    # plotting mean and pointwise CB
    plot!(p, x, post_mean, 
        ribbon = pointwise_CB, 
        label = "Posterior Mean", 
        fillalpha=.2,
        linecolor  = :black,
        fillcolor = :red)
    # plotting simultaneous CB
    plot!(p, x, [post_mean + joint_rect, post_mean - joint_rect], 
        linecolor  = :red,
        linestyle = :dash,
        label = "")
    
end

function matrixElement(x_vals, b1, b2, dt)
    return sum(b1.(x_vals) .* b2.(x_vals) * dt)
end

function vectorElement(x_vals, b)
    return sum(b.(x_vals[1:(length(x_vals)-1)]) .* diff(x_vals))
end    


function giraMatrix(path, basis)
   t_int = path.timeinterval
   x_vals = path.samplevalues
   
   dt = t_int[2] - t_int[1] 
    
   N =  length(basis)
   Sig = Array{Float64}(undef, N, N) 
    
   for i in 1:N
       for j in i:N
           #Sig[i,j] = sum(basis[i].(x_vals) .* basis[j].(x_vals) * dt)
           Sig[i,j] = matrixElement(x_vals, basis[i], basis[j], dt)
       end
   end
    
   Sig = Symmetric(Sig) 
    
   return Sig
end



function giraVector(path, basis)

   t_int = path.timeinterval
   x_vals = path.samplevalues
   
    
   N =  length(basis)
   m = Array{Float64}(undef, N) 
    
   for i in 1:N
       #m[i] = sum(basis[i].(x_vals[1:(length(x_vals)-1)]) .* diff(x_vals))
       m[i] = vectorElement(x_vals, basis[i])
   end
    
   return m
end

function likelog(G_vec, G_matrix; s = 1, alpha = 1.5)
    N = length(G_vec)
    Lambda = Diagonal([s^2 * k^(-2.0 * alpha - 1.0) for k in 1:N])
    
    B =  inv(Lambda) + G_matrix
    V = inv(B)
    
    return dot(G_vec, V * G_vec) - logdet(Lambda) - logdet(B)
    
end


function gira_empBayes(G_vec, G_matrix)
    return Optim.minimizer(optimize(x -> -likelog(G_vec, G_matrix, s = x[1], alpha = x[2]), [1.,1.])  )
end

function empBayes(path, basis)
    G_vec = giraVector(path, basis)
    G_matrix = giraMatrix(path, basis)
    return Optim.minimizer(optimize(x -> -likelog(G_vec, G_matrix, s = x[1], alpha = x[2]), [1.,1.])  )
end

function empBayes_fixscale(path, basis, s)
    G_vec = giraVector(path, basis)
    G_matrix = giraMatrix(path, basis)
    return optimize(x -> -likelog(G_vec, G_matrix, s = s, alpha = x), 0.05,5.0)
end


