# Plots posterior mean along with credible sets from posterior mean and variance of (theta(t_1), ..., theta(t_n))  
function post_plot(post_mean, post_var; a = 0.05, scale = true)
    pointwise_CB = point_set(post_var, p = 1 -a)
    joint_rect = simul_set(post_var, scale = scale, p = 1 - a)
    p =  plot(x, post_mean, 
            ribbon = pointwise_CB, 
            label = "Posterior Mean", 
            fillalpha=.2,
            linecolor  = :black,
            fillcolor = :red)
    plot!(p, x, [post_mean + joint_rect, post_mean - joint_rect], 
        linecolor  = :red,
        linestyle = :dash,
        label = "")
end

# Same plot but with a prior as input. Th specifies true theta.
function calc_plot(s, a, path, th)
    str = latexstring("\$ s = $(s), \\alpha = $(a) \$")
    post = post_from_data(mod, path, basis; alpha = a, s = s)
    pars = post_pars(post, x, basis)
    fig = post_plot(pars[1], pars[2])
        fig = plot!(fig, x, th, label = "True Drift", linecolor = :blue, title = str)
    return fig
end;
