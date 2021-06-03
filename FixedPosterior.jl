function matrixElement(x_vals, b1, b2, dt)
    return sum(b1.(x_vals) .* b2.(x_vals) * dt)
end

function vectorElement(x_vals, b)
    return sum(b.(x_vals[1:(length(x_vals)-1)]) .* diff(x_vals))
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
  
function giraMatrix(path, basis)
   t_int = path.timeinterval
   x_vals = path.samplevalues
   
   dt = t_int[2] - t_int[1] 
    
   N =  length(basis)
   Sig = Array{Float64}(undef, N, N) 
    
   for i in 1:N
       for j in i:N
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
       m[i] = vectorElement(x_vals, basis[i])
   end
    
   return m
end  

function prior_dist(s, alpha, basis)
    N = length(basis)
    d = GaussianVector(sparse(Diagonal([s * k^(-alpha -0.5) for k in 1:N])))
    
    return GaussianProcess(basis, d)
end  
  
function post_from_data(mod, path, basis; alpha = 0.7, s = 1.0)
    N = length(basis)
    
    prior = prior_dist(s, alpha, basis)
    
    return calculateposterior(prior, path, mod)
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
  
