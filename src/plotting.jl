using Gadfly

"""
    plotConvergenceCriterion(MCKL)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `MCKL::DenseVector`: Values of the KL divergence for each epoch.
"""
function plotConvergenceCriterion(MCKL::DenseVector)
    
    n_mckl = length(MCKL);

    plot(
        layer(x=1:n_mckl, y=MCKL, Geom.line),
        layer(x=1:n_mckl, y=MCKL, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Convergence criterion"),
        Guide.xlabel("Epoch"),
        Guide.ylabel("KL divergence"),
    )
end


"""
    plotConvergenceCriterion(MCKL, path)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `MCKL::DenseVector`: Values of the KL divergence for each epoch.
- `path::String`: Where to store the plot.
"""
function plotConvergenceCriterion(MCKL::DenseVector, path::String)
    
    n_epoch = length(MCKL);

    p = plot(
        layer(x=1:n_epoch, y=MCKL, Geom.line),
        layer(x=1:n_epoch, y=MCKL, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Convergence criterion"),
        Guide.xlabel("Epoch"),
        Guide.ylabel("KL divergence"),
    )

    Gadfly.draw(PNG(path, dpi=300), p)
end


"""
    plotTrace(model, hyperParam)

Plot evolution of a given hyper-parameter over CAVI iterations.

# Arguments
- `model::Model`: Model.
- `hyperParam::Symbol`: Hyper-parameter name as it is written in the model.
"""
function plotTrace(model::Model, hyperParam::Symbol)
    
    if isnothing(model.hyperParams[hyperParam].trace)
        error("Aucune trace n'est disponible !")
    end

    n_values = length(model.hyperParams[hyperParam].trace);
    values = model.hyperParams[hyperParam].trace;

    plot(
        layer(x=1:n_values, y=values, Geom.line),
        layer(x=1:n_values, y=values, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Trace de $hyperParam"),
        Guide.xlabel("Itérations"),
        Guide.ylabel("Valeur"),
    )
end


"""
    plotApproxVSMCMC(approxDensity, MCMCsample; a, b, xLabel, yLabel)

Plot the approximative density function against the sampled posterior density obtained by MCMC.

# Arguments
- `model::Model`: The model after CAVI has been run.
- `paramName::Symbol`: Parameter name as it is written in the model.
- `a::Real`: Left bound for the approximating marginal.
- `b::Real`: Right bound for the approximating marginal.
- `step::Real`: Precision of the approximating marginal plot.
"""
function plotApproxVSMCMC(
    model::Model,
    paramName::Symbol,
    a::Real,
    b::Real,
    step::Real,
)
    param = model.params[paramName];
    MCMCsample = param.mcmcSample[param.warmingSize:end];
    
    x = a:step:b;
    approxDensity(x::Real) = pdf(param.approxMarginal, x);

    plot(
        layer(x=x, y=approxDensity.(x), Geom.line, Theme(default_color="red")),
        layer(x=MCMCsample, Geom.histogram(density=true)),
        Guide.manual_color_key("Legend", ["Approximation", "Posteriori"], ["red", "deepskyblue"]),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC"),
        Guide.xlabel(String(paramName)),
        Guide.ylabel("Density"),
    )
end


"""
    plotApproxVSMCMC(approxDensity, MCMCsample, path; a, b, xLabel, yLabel)

Plot AND STORE the approximative density function against the sampled posterior density obtained by MCMC.

# Arguments
- `model::Model`: The model after CAVI has been run.
- `paramName::Symbol`: Parameter name as it is written in the model.
- `a::Real`: Left bound for the approximating marginal.
- `b::Real`: Right bound for the approximating marginal.
- `step::Real`: Precision of the approximating marginal plot.
- `path::String`: Where to store the plot.
"""
function plotApproxVSMCMC(
    model::Model,
    paramName::Symbol,
    a::Real,
    b::Real,
    step::Real,
    path::String,
)
    
    param = model.params[paramName];
    MCMCsample = param.mcmcSample[param.warmingSize:end];

    x = a:step:b;
    approxDensity(x::Real) = pdf(param.approxMarginal, x);

    plot(
        layer(x=x, y=approxDensity.(x), Geom.line, Theme(default_color="red")),
        layer(x=MCMCsample, Geom.histogram(density=true)),
        Guide.manual_color_key("Legend", ["Approximation", "Posteriori"], ["red", "deepskyblue"]),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC"),
        Guide.xlabel(String(paramName)),
        Guide.ylabel("Density"),
    )

    Gadfly.draw(PNG(path, dpi=300), p)
end