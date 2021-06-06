function figure4()
  ordata = DelimitedFiles.readdlm("butaneoutLang2Omega.txt", '\n',  ' ');
  # filtering the data
  data = vec(ordata)[1:1000:size(ordata, 1)]
  
  # time interval in picoseconds
  t_interval = 0.:1.:3999.;
  x = -pi:0.01:pi
  # quadratic variation process
  function quad_v(x)
      return pushfirst!(cumsum(diff(x) .^2), 0)
  end
  # estimate sigma
  q_vals = quad_v(data);
  # new units
  ps_per_unit = 4000 / q_vals[4000]
  t_interval = t_interval / ps_per_unit;
  path = SamplePath(t_interval,data);
  # defining model
  N = 150
  basis = [rad_fourier(k) for k in 1:N]
  
  mod = SDEModel(1., data[1], t_interval[length(t_interval)], t_interval[2]-t_interval[1]);
  
  ### PLOT 1: Quadratic Variation
  p1 =  plot(t_interval,  q_vals, label = "Observed Quadratic Variation", linecolor = :black)
  plot!(p1, [0, 1], [0, 1], seriestype = :straightline, label = "", linecolor = :red, linestyle = :dash)
  plot!(p1, xlab = "Time measured in units of u")
  
  ### PLOT 2: Histogram ###
  p2 = histogram(data, fill = 0.3, label = "Angle Observations", xlab = "Angle in Radians")
  
  ### PLOT 3: Time-Series plot ###
  p3 = plot(data, linecolor = :black, label = "Angle Observations", xlab = "Time in Nanoseconds")
  
  ### PLOT 4: Fixed prior estimation ###
  eta = 0.02
  alpha = 1.5
  post = post_from_data(mod, path, basis; alpha = alpha, s = 4 / sqrt(eta));
  pars = post_pars(post, x, basis)
  p4 = post_plot(pars[1], pars[2], a = 0.32, scale = true)
  plot!(p4, ylim = (-4,4))
    
  ### PLOT 4: EmpBayes prior ###
  (scale, alpha) = empBayes(path, basis);
  post = post_from_data(mod, path, basis; alpha = alpha, s = scale);
  pars = post_pars(post, x, basis)
  p4 = post_plot(pars[1], pars[2], a = 0.32, scale = true)
  plot!(p4, ylim = (-4,4))
    
  ### PLOT 5: Trace plot of s ###
  post = MCMC(path, rad_fourier, iter; j0 = 30, z0 = zeros(30), s_sq0 = 14.0, alpha = 1.5, A = 5/2, B = 5/2, C = log(0.95)); 
  f = func_from_coeffs(post[2], rad_fourier, x)[500:(iter+1),:];
  take = 500:3001
  rm = cumsum(sqrt.(post[3])) ./ (1:3001);
  p5 = plot(take, sqrt.(post[3])[take], xlab = "Iterations", label  = "s : Posterior Samples")
  plot!(p5, take, rm[take], label = "Running Mean", linewidth = 1.5)
  
  
  ### PLOT 6: Histogram J | X ###
  p6 = histogram(post[1][take], bins = 25,label = "", fill = 0.3, legend = :topleft, xlab = "J: Posterior Samples")
  
  ### PLOT 7: Trace plot of theta ###
  p7 = plot(xlab = "Iterations")
  plot!(p7, 500:(iter + 1), f[:,315], label = L"\theta(0)")
  plot!(p7, 500:(iter + 1), f[:,367], label = L"\theta(\pi / 6)")
  plot!(p7, 500:(iter + 1), f[:,472], label = L"\theta(\pi / 2)")
  
  ### PLOT 8: Estimation with Hierarchical Prior ###
  mu = vec(mean(f, dims = 1))
  band = sim_simul_band(f, mu; p = 0.68, marg = true);
  p8 = plot()
  plot!(p8, x, mu, ribbon = mu .- band[1], label = "", fill = 0.25, fillcolor = :red)
  plot!(p8, x, band[2], label = "", linecolor = :red, linestyle = :dash)
  plot!(p8, x, mu, label = "Posterior Mean", linecolor = :black, ylim = (-4,4))
  
  return (p1,p2,p3,p4,p5,p6,p7,p8)
end  
