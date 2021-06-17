# Calculates pointwise credible sets around posterior mean based on covariance matrix (sig).
function point_set(sig; p = 0.95)
   a = 1-p
   quantile(Normal(), 1 - a/2) * sqrt.(diag(sig))
end

# Calculates simultaneous credible sets around posterior mean by grid-approximation based on covariance matrix (sig).
function simul_set(sig; p = 0.95, N = 10^4, max = 50)
    sig = Symmetric(sig); d = MvNormal(sig)
    sample = rand(d,N)    
    rect = sqrt.(diag(sig)) / sqrt(sig[1,1]) 
    f(R) = mean(all(abs.(sample) .<= R * rect, dims = 1)) - p    
    R = find_zero(f, (0, max))    
    return R * rect
end

# Calculates credible margin around specified posterior mean (mu_vec) using samples (samps).   
function sim_simul_set(samps, mu_vec; p = 0.95, marg = false)
    N = size(samps, 1)
    dists = [maximum(abs.(samps[i, :] - mu_vec)) for i in 1:N]
    ranks = sortperm(dists)[1:ceil(Int, p * N)]
    used_samps = samps[ranks,:]
    res = hcat(vec(minimum(used_samps, dims =1)), vec(maximum(used_samps, dims =1)))        
    if marg    
    	a = 1 - p
    	obs = size(samps, 2)
    	res = (Transpose(hcat([quantile(samps[: , j], [a/2, 1 - a/2]) for j in 1:obs]...)), res)            
    end        
    return res    
end  
         
# Simulates from posterior (post) and calculates credible sets on interval (x) based on simulations        
function fixed_set(post, x; p = 0.95, N = 10^4, marg = false)
    samps = [rand(post).(x0) for i in 1:N, x0 in x]
    mu_vec = mean(post).(x)
    return sim_simul_set(samps,mu_vec, p = p, marg = marg)
end
            
# Calculates credible sets based on random empBayes samples (est_samples). For the posterior basis, model (mod), observed path (path) and interval (x) are specified.  
function emp_bayes_set(est_samples, basis, mod, path, x; p = 0.95, N = 10^4,  marg = false)
    samps = zeros(N, length(x))   
    sde = SDE(mu, mod)
    for i in 1:N 
        post = post_from_data(mod, path, basis, alpha = est_samples[i,2],  s = est_samples[i,1])
        samps[i, :] = rand(post).(x)
    end    
    return sim_simul_set(samps, zeros(length(x)), p = p, marg = marg)
end  
                  
                  
                  
# Check coverage of credible sets given a true parameter (theta) a prior (s, alpha) and a model (mod, sde).                          
function check_cov(theta, s, alpha, x, mod, sde, basis;  N = 10^4)    
    obs_theta = theta.(x)    
    sim_cov = zeros(N)
    point_cov = zeros(N, length(x))    
    for i in 1:N
        path = rand(sde)
        post = post_from_data(mod, path, basis, alpha = alpha, s = s)
        pars = post_pars(post, x, basis)
        sim_margin = simul_set(pars[2]); point_margin = point_set(pars[2])
        dist = abs.(obs_theta - pars[1]) 
        push!(sim_cov, all(dist .< sim_margin))
        point_cov[i,:] = dist .< point_margin
    end
    return (mean(point_cov, dims = 1), mean(sim_cov))
end  
