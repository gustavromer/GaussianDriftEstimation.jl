# Calculates pointwise confidence bands based on covariance matrix.
function point_band(sig; p = 0.95)
   a = 1-p
   quantile(Normal(), 1 - a/2) * sqrt.(diag(sig))
end

# Calculates simultaneous confidence bands by grid-approximation based on covariance matrix.
function simul_band(sig; p = 0.95, scale = true, N = 10^5, step = 0.01, max = 50)
    sig = Symmetric(sig)
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

# Calculates credible margin around specified posterior mean using samples.   
function sim_simul_band(samps, mu_vec; p = 0.95, marg = false)
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
        
# Simulates from posterior and calculates credible bands based on simulations.        
function fixed_band(post, x; p = 0.95, N = 10^3, marg = false)
    samps = [rand(post).(x0) for i in 1:N, x0 in x]
    mu_vec = mean(post).(x)
    return sim_simul_band(samps,mu_vec, p = p, marg = marg)
end 
          
# Calculates credible bands for empirical bayes using empirical distribution on (alpha, s).          
function emp_bayes_band(mu, basis, x, mod; p = 0.95, N = 10^3,  marg = false)
    samps = zeros(N, length(x))
    
    sde = SDE(mu, mod)
    or_path = rand(sde)
                  
    prog = Progress(N)
    Threads.@threads for i in 1:N
        path = rand(sde)
        
        bayes_est = empBayes(path, basis)
        
        post = post_from_data(mod, 
            or_path, 
            basis, 
            alpha = bayes_est[2],  
            s = bayes_est[1])

        
        samps[i, :] = rand(post).(x) - mean(post).(x)
        
         # update progress
        next!(prog)
    end
    
    return sim_simul_band(samps, zeros(length(x)), p = p, marg = marg)
end    
              
# Calculates coverage of credible bands from empirical bayes method. Empirical Bayes.               
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
