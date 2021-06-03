function figure3()
  # defining true drift
  theta(x) = sinpi(2x) - cospi(8x)
  # specify model (sig, x_0, T, dt) and sde
  model = SDEModel(1.0, 0.0, 2000., 0.01)
  sde = SDE(theta, model);
  path = rand(sde);
  iter = 2000;
  basis = [fourier(k) for k in 1:50]
  emp_pars = empBayes(path, [fourier(k) for k in 1:50]);
  a = 25
  b = 4;
  post = MCMC(path, unit_fourier, iter; j0 = 7, z0 = zeros(7), s_sq0 = emp_pars[1], alpha = emp_pars[2], A = a, B = b, C = log(0.95));
  x = 0:0.01:1
  f = func_from_coeffs(post[2], unit_fourier, x)[500:(iter+1),:];
  
  ### PLOT 1 : J | X^T ###
  p1 = histogram(post[1][500:(iter+1)], label = "J: Posterior Samples", xticks = 7:16)
  vline!(p1, [8.1], label = "True Number of Basis Functions", linecolor = :red, linestyle = :dash)
  
  ### PLOT 2  : Trace plot of s ###
  p2 = plot(500:(iter + 1), sqrt.(post[3])[500:(iter + 1)], xlab = "Iterations", label  = "s : Posterior Samples")
  rm = cumsum(sqrt.(post[3])) ./  collect(1.0:1.0:2001.0)
  plot!(p2, 500:(iter + 1), rm[500:(iter + 1)], label = "Running Mean", linewidth = 1.5)
  
  ### PLOT 3 : Trace plot of \theta ###
  p3 = plot()
  plot(p3, xlab = "Iterations")
  plot!(p3, 500:(iter + 1), f[500:(iter + 1),25], label = L"\theta(0.25)")
  plot!(p3, 500:(iter + 1), f[500:(iter + 1),35], label = L"\theta(0.35)")
  plot!(p3, 500:(iter + 1), f[500:(iter + 1),75], label = L"\theta(0.75)")
  plot!(p3, 500:(iter + 1), f[500:(iter + 1),100], label = L"\theta(1.00)")
  
  ### PLOT 4: Estimation and credible bands ###
  mu = vec(mean(f, dims = 1))
  band = sim_simul_band(f, mu; p = 0.95, marg = true);
  p4 = plot()
  plot!(p4, x, mu, ribbon = mu .- band[1], label = "", fill = 0.25, fillcolor = :red)
  plot!(p4, x, band[2], label = "", linecolor = :red, linestyle = :dash)
  plot!(p4, x, mu, label = "Posterior Mean", linecolor = :black)
  plot!(p4, x, theta, label = "True Drift", linecolor = :blue)
  
  ### PLOT 5: Estimation and credible bands - empBayes ###
  emp_post = post_from_data(model, path, basis; alpha = emp_pars[2], s = emp_pars[1])
  pars_from_emp = post_pars(emp_post, x, basis);
  p5 = post_plot(pars_from_emp[1], pars_from_emp[2])
  plot!(p5, x, theta, label = "True Drift", linecolor = :blue)
  
  ### PLOT 6: Posterior Simualtions - empBayes ###
  p6 = plot()
  for i in 1:10
      plot!(p6, x, rand(emp_post), label = "", linestyle = :dash)
  end  
  plot!(p6, x, theta, linecolor = :blue, label = "True Drift")
  
  ### PLOT 7: posterior simulations - MCMC###
  p5 = plot()
  for i in 1:10
    plot!(p5, x, f[1850-i,:], label = "", linestyle = :dash)
  end    
  plot!(p5, x, theta, linecolor = :blue, label = "True Drift")
end
