# Loading data and subsampling
ordata = DelimitedFiles.readdlm("butaneoutLang2Omega.txt", '\n',  ' ');
data = vec(ordata)[1:1000:size(ordata, 1)]; t_interval = 0.:1.:3999.; x = -pi:0.01:pi;

# Finding new unit using quadratic variation.
function quad_v(x) return pushfirst!(cumsum(diff(x) .^2), 0); end
q_vals = quad_v(data); ps_per_unit = 4000 / q_vals[4000]; t_interval = t_interval / ps_per_unit;
path = SamplePath(t_interval,data);

# Defining model
N = 50; basis = [rad_fourier(k) for k in 1:N]; 
mod = SDEModel(1., data[1], t_interval[length(t_interval)], t_interval[2]-t_interval[1]);

# Data plots
plot(t_interval,  q_vals, label = "Observed Quadratic Variation", linecolor = :black)
plot!([0, 1], [0, 1], seriestype = :straightline, label = "", linecolor = :red, linestyle = :dash)
plot!(xlab = "Time measured in units of u",size = (600, 300), dpi = 600, legend=:bottomright)
savefig("figures/sec5/fig4.png")
histogram(data, fill = 0.3, label = "Angle Observations", xlab = "Angle in Radians", size = (400, 300), dpi = 600)
savefig("figures/sec5/fig5.png")
plot(data, linecolor = :black, label = "Angle Observations", xlab = "Time in Nanoseconds",size = (400, 300), dpi = 600)
savefig("figures/sec5/fig6.png")

# Fixed prior
eta = 0.02; alpha = 1.5; 
post = post_from_data(mod, path, basis; alpha = alpha, s = 4 / sqrt(eta));
pars = post_pars(post, x, basis); post_plot(pars[1], pars[2], a = 0.32, scale = true);
plot!(ylim = (-4,4), size = (400, 300), dpi = 600, legend=:bottomright); savefig("figures/sec5/fig1.png")

# Empirical Bayes
(scale, alpha) = empBayes(path, basis); post = post_from_data(mod, path, basis; alpha = alpha, s = scale);
pars = post_pars(post, x, basis); post_plot(pars[1], pars[2], a = 0.32, scale = true,ylim = (-4,4))
plot!(size = (400, 300), dpi = 600, legend=:bottomright); savefig("figures/sec5/fig2.png")

# Hierarchical Bayes
iter = 3000; post = MCMC(path, rad_fourier, iter; j0 = 1, z0 = 0, s_sq0 = 1, alpha = 1.5, A = 5/2, B = 5/2, C = log(0.95)); 
f = func_from_coeffs(post[2], rad_fourier, x)[500:(iter+1),:]; take = 500:3001; rm = cumsum(sqrt.(post[3])) ./ (1:3001);

# Traceplots
plot(take, sqrt.(post[3])[take], xlab = "Iterations", label  = "s : Posterior Samples"); 
plot!(take, rm[take], label = "Running Mean", linewidth = 1.5,size = (400, 300), dpi = 600,legend = :topleft)
savefig("figures/sec5/fig7.png")
histogram(post[1][take], bins = 25,label = "", fill = 0.3, legend = :topleft, xlab = "J: Posterior Samples",size = (400, 300), dpi = 600))
savefig("figures/sec5/fig8.png")
plot(xlab = "Iterations", size = (400, 300), dpi = 600,legend = :topleft)
for i in [315, 367, 472] plot!(1:size(f,1), f[:,315], label = string(-pi + i / 100)) end
savefig("figures/sec5/fig9.png")
  
# Credible sets and posterior mean based on samples
mu = vec(mean(f, dims = 1)), band = sim_simul_band(f, mu; p = 0.68, marg = true);
plot(x, mu, ribbon = mu .- band[1], label = "", fill = 0.25, fillcolor = :red), plot!(x, band[2], label = "", linecolor = :red, linestyle = :dash)
plot!(x, mu, label = "Posterior Mean", linecolor = :black, ylim = (-4,4),size = (400, 300), dpi = 600, legend=:bottomright)
savefig("figures/sec5/fig3.png")
  
### Figures are saved in a 'figures/sec5' folder ###   
