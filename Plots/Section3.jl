# Define true drift
function a(x)
    if x > 2/3
        return 2/7 + (2/7)x
    else
        return 2 / 7 - x - 2/7 * (1. - 3x)sqrt(abs(1. - 3x))
    end
end
theta(x) = 12. * (a(mod(x,1.0)) + 0.05);
    
# specify model
N = 50; basis = [unit_fourier(k) for k in 0:(N-1)]
x = 0:0.01:1; model = SDEModel(1.0, 0.0, 1000., 0.01); sde = SDE(theta, model);
    
# Sample empirical bayes estimator
n_obs = 500; ests = zeros(n_obs, 2); 
for i in 1:n_obs path = rand(sde); ests[i,:] = empBayes(path, basis); end
histogram(ests[:,2], bins = 20,label = "", xlab = L"\hat{\alpha}", alpha = 0.65)
plot!([1.5], seriestype="vline",linestyle = :dash, linewidth = 1.5, label = "True Holder Exponent", legend=:topleft,linecolor = :red,size = (450, 350), dpi = 600)
savefig("figures/sec3/fig1.png")
scatter(ests[:,2], ests[:,1], xlab = L"\hat{\alpha}",ylab = L"\hat{s}", label = "",size = (450, 350), dpi = 600); savefig("figures/sec3/fig2.png");

# Plot different posterior means for different priors
n_curves = 20; pars = [(1, 2.5), (1, 0.5), (20, 2)];
plots = [plot(), plot(), plot(), plot()];
for i=1:n_curves, j=1:4
    path = rand(sde);
    if j == 4 (s, a) = empBayes(path, basis) else (s, a) = pars[j] end
    post = post_from_data(model, path, basis, s = s_emp, alpha = a_emp)
    plot!(plots[j], x, rand(post), linestyle = :dash, label = "")
    end
end;
for j in 1:4
    plot(plots[j], size = (450, 300), dpi = 600, legend=:bottomright);
    savefig("figures/sec3/fig" * string(j+3) * ".png");
end

# Look at specific sample path and find posterior
path = rand(sde); emp_ests = empBayes(path, basis);
emp_post = post_from_data(model, path, basis; alpha = emp_ests[2], s = emp_ests[1]); emp_pars = post_pars(emp_post, x, basis);
u_fixed = fixed_band(emp_post, x; p = 0.95, N = 10^4, marg = true); u_calc = simul_band(emp_pars[2]);

# Credible set based on fixed empirical Bayes estimator and random estimator samples 
u = emp_bayes_band(ests, basis, model, path, x);
plot(x, emp_pars[1] .+ u[2][:,1], linecolor = :red, linestyle = :dash, label = "Random Prior: Simultaneous Band")
plot!(x, emp_pars[1] .+ u[2][:,2], linecolor = :red, linestyle = :dash, label = "")
plot!(x, u_fixed[2][:,1], linecolor = :blue, label = "", linestyle = :dash)
plot!(x, u_fixed[2][:,2], linecolor = :blue, label = "Fixed Prior: Simultaneous Band", linestyle = :dash)
plot!(x, emp_pars[1], linecolor = :black, label = "Posterior Mean")
plot!(x, theta, linecolor = :blue, label = "True Drift",size = (450, 300), dpi = 600, legend=:topright)
savefig("figures/sec3/fig8.png")
plot(x, emp_pars[1] .+ u[1][:,1], linecolor = :red, linestyle = :dash, label = "Random Prior: Pointwise Band")
plot!(x, emp_pars[1] .+ u[1][:,2], linecolor = :red, linestyle = :dash, label = "")
plot!(x, u_fixed[1][:,1], linecolor = :blue, label = "", linestyle = :dash)
plot!(x, u_fixed[1][:,2], linecolor = :blue, label = "Fixed Prior: Pointwise Band", linestyle = :dash)
plot!(x, emp_pars[1], linecolor = :black, label = "Posterior Mean")
plot!(x, theta, linecolor = :blue, label = "True Drift",size = (450, 300), dpi = 600, legend=:topright)
savefig("figures/sec3/fig9.png")

# Checking coverage of empirical bayes
H = check_cov(theta, emp_ests[1], emp_ests[2], x, model, sde, basis,  N = 500)
plot(x, vec(H[1]), label = "Frequentist Coverage")
hline!(x, [0.95], linestyle = :dash, linecolor = :black, label = "Bayesian Posterior Coverage",ylims = (0.875,1.0))
plot!(size = (450, 300), dpi = 600, legend=:bottomleft); savefig("figures/sec3/fig10.png");

# Checking coverage for other fixed priors.
point_plot = plot(); loop = [(1.0,2.5), (1.0, 0.5), (20.0, 2.0)]
for i in 1:3
    s = loop[i][1]
    al = loop[i][2]
    int_s = check_cov(theta, s, al, x, model, sde, basis, N = 500)
    plot!(x, vec(int_s[1]), label = latexstring("\$ s = $(s), \\alpha = $(al) \$"))
end
hline!([0.95], label = "Bayesian Posterior Coverage", linestyle = :dash, linecolor = :black);
plot(point_plot, legend = :bottomleft, ylim = (0.875,1),size = (450, 300), dpi = 600)
savefig("figures/sec3/fig11.png")

    
### Figures are saved in a 'figures/sec3' folder ###      
