# Define model
theta(x) = sinpi(2x) - cospi(8x); model = SDEModel(1.0, 0.0, 2000., 0.01)
sde = SDE(theta, model); path = rand(sde); iter = 3000; x = 0:0.01:1
basis = [fourier(k) for k in 0:49]; 

# Empirical Bayes estimator
emp_pars = empBayes(path, [fourier(k) for k in 1:49]);

# Hierarchical Bayes posterior
post = MCMC(path, unit_fourier, iter; j0 = 1, z0 = 0, s_sq0 = 1, alpha = 1.5, A = 4.0, B = 25.0, C = log(0.95));

# Implied sample paths
f = func_from_coeffs(post[2], unit_fourier, x)[500:(iter+1),:];

# Trace plots
histogram(post[1][500:(iter+1)].-1, label = "J: Posterior Samples", xticks = 7:16)
vline!([8.1], label = "True Number of Basis Functions", linecolor = :red, linestyle = :dash, size = (650, 350), dpi = 600)
savefig("figures/sec4/fig1.png")
plot(500:(iter + 1), sqrt.(post[3])[500:(iter + 1)], xlab = "Iterations", label  = "s : Posterior Samples")
rm = cumsum(sqrt.(post[3])) ./  collect(1.0:1.0:3001.0)
plot!(500:(iter + 1), rm[500:(iter + 1)], label = "Running Mean", linewidth = 1.5)
plot!(size = (450, 300), dpi = 600)
savefig("figures/sec4/fig2.png")
plot(xlab = "Iterations")
for i in [25,35,75,100] plot!(1:size(f,1) f[,i], label = string(i / 100) end
savefig("figures/sec4/fig3.png")
  
# Posterior mean and credible bands based on sumulation
mu = vec(mean(f, dims = 1)); band = sim_simul_band(f, mu; p = 0.95, marg = true);
plot(x, mu, ribbon = mu .- band[1], label = "", fill = 0.25, fillcolor = :red)
plot!(x, band[2], label = "", linecolor = :red, linestyle = :dash)
plot!(x, mu, label = "Posterior Mean", linecolor = :black)
plot!(x, theta, label = "True Drift", linecolor = :blue, size = (450, 300), dpi = 600);savefig("figures/sec4/fig4.png");
  
# Empirical Bayes posterior for comparison
emp_post = post_from_data(model, path, basis; alpha = emp_pars[2], s = emp_pars[1])
pars_from_emp = post_pars(emp_post, x, basis); post_plot(pars_from_emp[1], pars_from_emp[2])
plot!(x, theta, label = "True Drift", linecolor = :blue, size = (450, 300), dpi = 600)
savefig("figures/sec4/fig5.png")
plot()
  
# Random posterior sample paths
for i in 1:10 plot!(x, rand(emp_post), label = "", linestyle = :dash) end  
plot!(x, theta, linecolor = :blue, label = "True Drift", size = (450, 300), dpi = 600); savefig("figures/sec4/fig6.png");
plot()
for i in 1:10 plot!(g, x, f[2141-i,:], label = "", linestyle = :dash) end 
plot!(x, theta, linecolor = :blue, label = "True Drift",size = (450, 300), dpi = 600)
savefig("figures/sec4/fig7.png")

### Figures are saved in a 'figures/sec4' folder ###   
