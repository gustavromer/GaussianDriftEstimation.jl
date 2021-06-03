# Section 1

function figure1()
  # defining true drift
  theta(t) = sinpi(2t) - cospi(8t)

  # defining basis
  N = 50
  basis = pushfirst!([fourier(k) for k in 1:(N-1)], x -> 1.0)
  
  
  # specify model (sig, x_0, T, dt) and sde
  mod = SDEModel(1.0, 0.0, 2000., 0.01)
  sde = SDE(theta, mod)
  
  # simulating path
  path  = rand(sde);
  
  #### PLOT1 : sample path plot ####
  p1 = plot(path.timeinterval, path.samplevalues, 
    label = "Observed Sample Path",
    linecolor = "black")
  
  
  # calculating posterior
  post = post_from_data(mod, path, basis; alpha = 0.7)
  # plotting interval
  x = 0:0.01:1
  # getting fidi
  pars = post_pars(post, x, basis)
  post_plot(pars[1], pars[2])
  
  #### PLOT2 : Post-mean and credible bands ####
  p2 = plot!(x, theta, label = "True Drift", linecolor = :blue)
  
  # Simulate from posterior check coverage
  pointwise_CB = point_band(pars[2]);
  joint_rect = simul_band(pars[2], scale = true);
  
  #### PLOT3 : Posterior Simulations ####
  p3 = plot()
  plot!(p3,x, [rand(post) for k in 1:10], label = "")
  plot!(p3,x, pars[1], ribbon = pointwise_CB,
    label = "", 
    fillalpha=0.2,
    linecolor = nothing,
    fillcolor = :red)
  plot!(p3, x, [pars[1] .+ joint_rect, pars[1] .- joint_rect], 
    linecolor  = :red,
    linestyle = :dash,
    label = "")
  
  # simulated credibility bands
  sim_joint_rect = fixed_band(post, x, N = 10^4, marg = true);
  p4 = plot()

  plot!(p4, x, -joint_rect, label = "", linecolor = :blue, linestyle = :dash)
  plot!(p4,x, joint_rect, label = "Simultaneous Margin", linecolor = :blue, linestyle = :dash)
  plot!(p4,x, -pointwise_CB, label = "Pointwise Margin", linecolor = :blue, linestyle = :dashdotdot)
  plot!(p4,x, pointwise_CB, label = "", linecolor = :blue, linestyle = :dashdotdot)
  
  
  plot!(p4,x , sim_joint_rect[2][:,1] - pars[1], label = "Simulated Simultaneous Margin", linecolor = :red, linestyle = :dash)
  plot!(p4,x , sim_joint_rect[2][:,2] - pars[1], label = "", linecolor = :red, linestyle = :dash)
  
  
  plot!(p4,x , sim_joint_rect[1][:,1] - pars[1], label = "Simulated Pointwise Margin", linecolor = :red, linestyle = :dashdotdot)
  plot!(p4,x , sim_joint_rect[1][:,2] - pars[1], label = "", linecolor = :red, linestyle = :dashdotdot)
  
  #### PLOT4 : Comparison of simulation and grid approach ####
  plot!(x, x -> 0.,linecolor  = :black,
    linestyle = :solid,label = "", legend=:inside)
  
  ### PLOT5 (from section 2) : Frequentist coverage ###
  coverage_check = check_cov(theta, 1.0, 0.7, x, mod, sde, basis,  N = 500)
  p5 = plot()
  plot!(p5, x, vec(coverage_check[1]), label = "Frequentist Coverage",ylims = (0.875,1.0))
  hline!(p5, [0.95], label = "Bayesian Posterior Coverage", linestyle = :dash, linecolor = :black,
  legend = :bottomleft)
  
  return (p1,p2,p3,p4,p5)
end  

