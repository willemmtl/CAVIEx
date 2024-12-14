using Optim, Distributions, Gadfly

include("iGMRF.jl");
include("utils.jl");
include("models/AbstractModel.jl");
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
    MCKL = Float64[];
    Hθ = Hθ₀;
    push!(MCKL, MonteCarloKL(model))
    
    ### ---- Ascent ---- ###
    duration = @elapsed begin
        for _ = 1:n_epoch
            push!(MCKL, MonteCarloKL(model))
            for _ = 1:epoch_size
                updateHyperParams!(model);
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
function MonteCarloKL(model::AbstractModel.Model)
    
    N = 1000;
    supp = zeros(model.nHyperParams, N);

    # Unpacking
    Hθ = AbstractModel.getHθ(model);
    
    for (i, f) in enumerate(model.logApproxMarginals)
        supp[i, :] = rand(f(Hθ), N);
    end
    
    logTarget = evaluateLogMvDensity(x -> model.logTargetDensity(x), supp);
    logApprox = evaluateLogMvDensity(x -> model.logApproxDensity(x), supp);
    
    return sum(logApprox .- logTarget) / N
end;


"""
    updateHyperParams!(model)

Update approximation parameters.

Correspond to one iteration in an epoch of the CAVI algorithm.

# Arguments :
- `model::AbstractModel.BaseModel`: TBD.
"""
function updateHyperParams!(model::AbstractModel.Model)
    
    for (k, f) in enumerate(model.refiningFunctions) 
        Hθ[k] = f(Hθ);
    end
    
end;