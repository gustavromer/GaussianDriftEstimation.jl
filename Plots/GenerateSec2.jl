
function figure2()
    
  ### PLOT1 : Empirical distribution of alpha ###  
  
  # defining drift
  function a(x)
    if x > 2/3
        res = -2/7 + (2/7)x
    else
        res = 2 / 7 - x - 2/7 * (1. - 3x)sqrt(abs(1. - 3x))
    end
    return res
  end

  theta(x) = 12. * (a(mod(x,1.0)) + 0.05);
  # defining basis
  N = 50
  basis = pushfirst!([fourier(k) for k in 1:(N-1)], x -> 1.0)
  x = 0:0.01:1
  # specify model (sig, x_0, T, dt) and sde
  model = SDEModel(1.0, 0.0, 1000., 0.01)
  sde = SDE(theta, model);
  n_obs = 1000
  ests = zeros(n_obs, 2) 
  
  prog = Progress(n_obs)
  Threads.@threads for i in 1:n_obs
      # sample random path
      path = rand(sde)
      # calculate empBayes estimators
      ests[i,:] = empBayes(path, basis)
      
      # update progress
      next!(prog)
  end

  p1 = histogram(ests[:,2], 
    bins = 20, 
    label = "", 
    xlab = L"\hat{\alpha}",
   alpha = 0.65)
  plot!(p1, [1.5], seriestype="vline", 
    linestyle = :dash,
    linewidth = 1.5,
    label = "True Hölder Exponent",
    legend=:topleft,
    linecolor = :red)  
    
    ### PLOT2 : Scatterplot of (s,alpha) ###  
    p2 = scatter(ests[:,2], ests[:,1], xlab = L"\hat{\alpha}",ylab = L"\hat{s}", label = "")
    
    ### PLOT3 : Hyper-parameter sensitivity
    
    n_curves = 20
    (s_over, a_over) = (1, 2.5)
    (s_under, a_under) = (1, 0.5)
    (s_scaled, a_scaled) = (20, 2);
    
    emp_bayes_ests = plot()
    over_smooth_ests = plot()
    under_smooth_ests = plot()
    scaled_ests = plot()

    prog = Progress(n_curves, dt = 0.2)
    Threads.@threads for i in 1:(4 * n_curves)
      
      # Sample path from model
      path = rand(sde)
      
      if(Threads.threadid() == 1)
          # Empirical Bayes
          (s_emp, a_emp) = empBayes(path, basis)
          post_emp = post_from_data(model, path, basis, s = s_emp, alpha = a_emp)
          plot!(emp_bayes_ests, x, mean(post_emp), linestyle = :dash, label = "")
      elseif(Threads.threadid() == 2)
          # Over Smooth
          post_over = post_from_data(model, path, basis, s = s_over, alpha = a_over)
          plot!(over_smooth_ests, x, mean(post_over), linestyle = :dash, label = "")
      elseif(Threads.threadid() == 3)
          # Under Smooth
          post_under = post_from_data(model, path, basis, s = s_under, alpha = a_under)
          plot!(under_smooth_ests, x, mean(post_under), linestyle = :dash, label = "")
      else
           # Scaled
          post_scaled = post_from_data(model, path, basis, s = s_scaled, alpha = a_scaled)
          plot!(scaled_ests, x, mean(post_scaled), linestyle = :dash, label = "")
      end    
      
      # updating progress
      next!(prog)
  end;
    
  ### PLOT4 : Simulatenous bands  
  path = rand(sde)
  emp_ests = empBayes(path, basis)
  emp_post = post_from_data(model, path, basis; alpha = emp_ests[2], s = emp_ests[1])
  emp_pars = post_pars(emp_post, x, basis);  
  u_fixed = fixed_band(emp_post, x; p = 0.95, N = 10^3, marg = true)
  u_calc = simul_band(emp_pars[2]);  
  u = emp_bayes_band(mean(emp_post), basis, x, model; p = 0.95, N = 750, marg = true);  
  p4 = plot()
  plot!(p4, x, emp_pars[1] .+ u[2][:,1], linecolor = :red, linestyle = :dash, label = "Random Prior: Simultaneous Band")
  plot!(p4, x, emp_pars[1] .+ u[2][:,2], linecolor = :red, linestyle = :dash, label = "")
  plot!(p4, x, u_fixed[2][:,1], linecolor = :blue, label = "", linestyle = :dash)
  plot!(p4, x, u_fixed[2][:,2], linecolor = :blue, label = "Fixed Prior: Simultaneous Band", linestyle = :dash)
  
  
  plot!(p4, x, emp_pars[1], linecolor = :black, label = "Posterior Mean")
  plot!(p4, x, theta, linecolor = :blue, label = "True Drift")  
    
  ### PLOT 5 : Pointwise Bands
  p5 = plot()

  plot!(p5, x, emp_pars[1] .+ u[1][:,1], linecolor = :red, linestyle = :dash, label = "Random Prior: Pointwise Band")
  plot!(p5, x, emp_pars[1] .+ u[1][:,2], linecolor = :red, linestyle = :dash, label = "")
  plot!(p5, x, u_fixed[1][:,1], linecolor = :blue, label = "", linestyle = :dash)
  plot!(p5, x, u_fixed[1][:,2], linecolor = :blue, label = "Fixed Prior: Pointwise Band", linestyle = :dash)

  plot!(p5, x, emp_pars[1], linecolor = :black, label = "Posterior Mean")
  plot!(p5, x, theta, linecolor = :blue, label = "True Drift")  
  
 ### PLOT  6 : Frequentist Coverage   
  H = check_cov(theta, emp_ests[1], emp_ests[2], x, model, sde, basis,  N = 250)
  p6 = plot()
  plot!(p6, x, vec(H[1]), label = "Frequentist Coverage")
  hline!(p6, x, [0.95], linestyle = :dash, linecolor = :black, label = "Bayesian Posterior Coverage",ylims = (0.875,1.0))

  return (p1,p2,emp_bayes_ests, over_smooth_ests, under_smooth_ests, scaled_ests, p4, p5, p6) 
end  
  
  
  
  
