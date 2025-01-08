using Distributions

include("utils.jl");
include("models/Model.jl");

"""
    runCAVI(n_epoch, epoch_size, hp0, model)

Run the CAVI algorithm over the given model.

Initial values must be entered by the user.

# Arguments
- `n_epoch::Integer`: Number of epochs to compute i.e. number of type we'll compute the convergence criterion.
- `epoch_size::Integer`: Size of each epoch i.e. number of iterations before computing the convergence criterion.
- `hp0::DenseVector`: Initial values of the hyper-parameters.
- `model::Model`: Model's configuration.
"""
function runCAVI(n_epoch::Integer, epoch_size::Integer, hp0::DenseVector, model::Model)

    ### ---- Basic checks ---- ###
    checkIfClear(model);

    ### ---- Initialization ---- ###
    initialize!(model, hp0);
    MCKL = Float64[];
    push!(MCKL, MonteCarloKL(model))
    
    ### ---- Ascent ---- ###
    duration = @elapsed begin
        for _ = 1:n_epoch
            for _ = 1:epoch_size
                refineHyperParams!(model);
            end
            push!(MCKL, MonteCarloKL(model))
        end
    end
    println("Minimization made in ", duration, " s")
    
    return MCKL
end;


"""
    checkIfClear(model)

Check whether the model is clear before running CAVI.

# Arguments
- `model::Model`: Model's configuration.
"""
function checkIfClear(model::Model)

    n_trace = 0;
    for (_, hyperParam) in model.hyperParams
        if !isnothing(hyperParam.trace)
            n_trace += 1;
        end
    end

    if n_trace != 0
        @warn "Model may have not been cleared : $n_trace traces are not empty."
    end

end


"""
    MonteCarloKL(model)

Compute the convergence criterion with current hyper-parameters.

# Arguments
- `model::Model`: Model's configuration.
"""
function MonteCarloKL(model::Model)
    
    syncMarginals!(model);

    N = 1000;
    supp = generateApproxSample(model, N);

    Hθ = getHpValues(model);

    logTarget = evaluateLogMvDensity(x -> model.logTargetDensity(x), supp);
    logApprox = evaluateLogMvDensity(x -> model.logApproxDensity(x, Hθ), supp);
    
    return sum(logApprox .- logTarget) / N
end;


"""
    refineHyperParams!(model)

Update approximation parameters.

Correspond to one iteration in an epoch of the CAVI algorithm.

# Arguments :
- `model::Model`: Model's configuration.
"""
function refineHyperParams!(model::Model)

    for (_, hyperParam) in model.hyperParams
        hpValues = getHpValues(model);
        push!(hyperParam.trace, hyperParam.refiningFunction(hpValues));
    end
    
end;