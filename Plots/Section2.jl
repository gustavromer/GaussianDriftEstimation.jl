# Specify models
theta(t) = sinpi(2t) - cospi(8t); theta_2(t) = sinpi(2t), 
x = 0:0.01:1; 
basis = [unit_fourier(k) for k in 0:49]
mod = SDEModel(1.0, 0.0, 2000., 0.01); sde = SDE(theta, mod);

#simulate sample path
path  = rand(sde);
plot(path.timeinterval, path.samplevalues, label = "Observed Sample Path", linecolor = "black", size = (400, 300), dpi = 600)
savefig("figures/sec2/fig4.png")

# calculate posterior
post = post_from_data(mod, path, basis; alpha = 0.75); pars = post_pars(post, x, basis);
post_plot(pars[1], pars[2])
plot!(x, theta, label = "True Drift", linecolor = :blue, size = (400, 300), dpi = 600); savefig("figures/sec2/fig3.png")
plot(x, [rand(post) for k in 1:10], label = "",size = (400, 300), dpi = 600, legend=:bottomright); savefig("figures/sec2/fig5.png")

# calculate grid/exact sets
pointwise_CB = point_band(pars[2]); joint_rect = simul_band(pars[2], scale = true); 
sim_joint_rect = fixed_band(post, x, N = 10^4, marg = true);
plot(x, -joint_rect, label = "", linecolor = :blue, linestyle = :dash)
plot!(x, joint_rect, label = "Simultaneous Band", linecolor = :blue, linestyle = :dash)
plot!(x, -pointwise_CB, label = "Pointwise Band", linecolor = :blue, linestyle = :dashdotdot)
plot!(x, pointwise_CB, label = "", linecolor = :blue, linestyle = :dashdotdot)
plot!(x , sim_joint_rect[2][:,1] - pars[1], label = "Simulated Simultaneous Band", linecolor = :red, linestyle = :dash)
plot!(x , sim_joint_rect[2][:,2] - pars[1], label = "", linecolor = :red, linestyle = :dash)
plot!(x , sim_joint_rect[1][:,1] - pars[1], label = "Simulated Pointwise Band", linecolor = :red, linestyle = :dashdotdot)
plot!(x , sim_joint_rect[1][:,2] - pars[1], label = "", linecolor = :red, linestyle = :dashdotdot)
plot!(x, x -> 0.,linecolor  = :black, linestyle = :solid,label = "Posterior Mean Reference", legend=:inside, size = (400, 300), dpi = 600)
savefig("figures/sec2/fig6.png")

# Check coverage of sets
coverage_check = check_cov(theta, 1.0, 0.50, x, mod, sde, basis,  N = 10^3)
freq_point = plot(x, vec(coverage_check[1]), label = "Frequentist Coverage",ylims = (0.875,1.0))
hline!(freq_point, [0.95], label = "Bayesian Posterior Coverage", linestyle = :dash, linecolor = :black,
legend = :bottomleft)
plot!(size = (450, 300), dpi = 600)
savefig("figures/sec2/fig7.png")

# Try out different priors
s_vec = [0.05, 0.1, 100]; alpha_vec = [0.5, 1.0, 1.5];
sde = SDE(theta, mod); sde_2 = SDE(theta_2, mod), path = rand(sde); path_2 = rand(sde_2)
p1 = []; p2 = [];
for a in alpha_vec
    push!(p1, calc_plot(1.0, a, path, theta))
    push!(p2, calc_plot(1.0, a, path_2, theta_2))
end
for s in s_vec
    push!(p1, calc_plot(s, 1.5, path, theta))
    push!(p2, calc_plot(s, 1.5, path_2, theta_2))
end
plot(p1..., layout = (2,3),size = (1062.5, 500), dpi = 600); savefig("figures/sec2/fig8.png");
plot(p2..., layout = (2,3),size = (1062.5, 500), dpi = 600); savefig("figures/sec2/fig9.png");
        
# Check their coverage
covs = [];s_obs = [];point_plot = plot()
for a in [0.5, 0.75, 1.0, 1.5]
    int_s = check_cov(theta, 1.0, a, x, mod, sde, basis, N = 10^3)
    plot!(x, vec(int_s[1]), label = latexstring("\$ \\alpha = $(a) \$"))
    push!(s_obs, int_s[2])
end
hline!([0.95], label = "Posterior Coverage", linestyle = :dash, linecolor = :black);
plot(point_plot,size = (450, 300), dpi = 600, legend = :bottomleft); savefig("figures/sec2/fig10.png");
plot([0.5, 0.75, 1.0, 1.5], s_obs, xlab = L"\alpha", label = "Frequentist Coverage")
scatter!([0.5, 0.75, 1.0, 1.5], s_obs, label = ""); hline!([0.95], label = "Posterior Coverage", linestyle = :dash, linecolor = :black,size = (450, 300), dpi = 600);
savefig("figures/sec2/fig11.png")
