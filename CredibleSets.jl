# Calculates pointwise confidence sets based on covariance matrix.
function point_set(sig; p = 0.95)
   a = 1-p
   quantile(Normal(), 1 - a/2) * sqrt.(diag(sig))
end
# Calculates simultaneous confidence sets by grid-approximation based on covariance matrix.
function simul_set(sig; p = 0.95, N = 10^4, step = 0.01, max = 50)
    sig = Symmetric(sig); d = MvNormal(sig)
    sample = rand(d,N)    
    rect = sqrt.(diag(sig)) / sqrt(sig[1,1]) 
    f(R) = mean(all(abs.(sample) .<= R * rect, dims = 1)) - p    
    R = find_zero(f, (0, max))    
    return R * rect
end
# Calculates credible margin around specified posterior mean using samples.   
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
# Simulates from posterior and calculates credible sets based on simulations.        
function fixed_set(post, x; p = 0.95, N = 10^4, marg = false)
    samps = [rand(post).(x0) for i in 1:N, x0 in x]
    mu_vec = mean(post).(x)
    return sim_simul_set(samps,mu_vec, p = p, marg = marg)
end
# Calculates credible sets based on random (alpha, s) samples from empirical Bayes.   
function emp_bayes_set(mu, basis, x, mod; p = 0.95, N = 10^4,  marg = false)
    samps = zeros(N, length(x))   
    sde = SDE(mu, mod)
    or_path = rand(sde)
    for i in 1:N
        bayes_est = empBayes(rand(sde), basis)    
        post = post_from_data(mod, or_path,  basis, alpha = bayes_est[2],  s = bayes_est[1])
        samps[i, :] = rand(post).(x) - mean(post).(x)
    end    
    return sim_simul_set(samps, zeros(length(x)), p = p, marg = marg)
end     
# Check coverage of credible sets given a true parameter and a prior.                          
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
