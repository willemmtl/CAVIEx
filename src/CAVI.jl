using Optim, Distributions

include("utils.jl");

if !isdefined(Main, :AbstractModel)
    include("models/AbstractModel.jl");
end
using .AbstractModel


"""
    runCAVI(n_epoch, epoch_size, Hθ₀, model)

Run the CAVI algorithm over the given model.

Initial values must be entered by the user.

# Arguments :
- `n_epoch::Integer`: Number of epochs to compute i.e. number of type we'll compute the convergence criterion.
- `epoch_size::Integer`: Size of each epoch i.e. number of iterations before computing the convergence criterion.
- `Hθ₀::DenseVector`: Initial values of the hyper-parameters.
- `model::AbstractModel.BaseModel`: Object containing the target, approximation and updating functions and data.
"""
function runCAVI(n_epoch::Integer, epoch_size::Integer, Hθ₀::DenseVector, model::AbstractModel.BaseModel)

    ### ---- Initialization ---- ###
    model.hyperParamsValue .= Hθ₀;
    AbstractModel.storeValues(model)
    MCKL = Float64[];
    push!(MCKL, MonteCarloKL(model))
    
    ### ---- Ascent ---- ###
    duration = @elapsed begin
        for _ = 1:n_epoch
            push!(MCKL, MonteCarloKL(model))
            for _ = 1:epoch_size
                refineHyperParams!(model);
            end
        end
    end
    println("Minimization made in ", duration, " s")
    
    return MCKL, Hθ
end;


"""
    MonteCarloKL(model)

Compute the convergence criterion with current hyper-parameters.

# Arguments :
- `model::AbstractModel.BaseModel`: TBD.
"""
function MonteCarloKL(model::AbstractModel.BaseModel)
    
    N = 10000;
    supp = zeros(model.nHyperParams, N);

    for (i, marginal) in enumerate(model.approxMarginals)
        supp[i, :] = rand(marginal(model.hyperParamsValue), N);
    end

    logTarget = evaluateLogMvDensity(x -> model.logTargetDensity(x), supp);
    logApprox = evaluateLogMvDensity(x -> model.logApproxDensity(x, model.hyperParamsValue), supp);
    
    return sum(logApprox .- logTarget) / N
end;


"""
    refineHyperParams!(model)

Update approximation parameters.

Correspond to one iteration in an epoch of the CAVI algorithm.

# Arguments :
- `model::AbstractModel.BaseModel`: TBD.
"""
function refineHyperParams!(model::AbstractModel.BaseModel)

    for (k, f) in enumerate(model.refiningFunctions)
        model.hyperParamsValue[k] = f(model.hyperParamsValue);
    end
    
    AbstractModel.storeValues(model);

end;