module AbstractModel

abstract type Model end

"""
    BaseModel

Sets the context for the variational algorithm.

# Attributes :
- `nHyperParams::Integer`: Number of hyper-parameters of the approximating function.
- `Hθ::DenseVector`: Hyper-parameters.
- `logTargetDensity::Function`: Density to approximate (log).
    MUST have the shape θ -> f(θ) where θ is the vectors of parameters.
- `logApproxDensity::Function`: Approximating density (log).
    CORRESPONDS to the sum of the log-approx-marginals (mean field).
- `logApproxMarginals::Vector{Function}`: Approximating marginal densities for each hyper-parameter (log).
    MUST have the shape (θ, Hθ) -> f(θ, Hθ) where θ and Hθ are respectively the vectors of parameters and hyper-parameters.
- `refiningFunctions::Vector{Function}`: Functions used to update hyper-parameters in the CAVI algorithm.
    MUST have the shape (Hθ) -> f(Hθ) where Hθ is the vector of hyper-parameters.
"""
struct BaseModel
    nHyperParams::Integer
    Hθ::DenseMatrix
    logTargetDensity::Function
    logApproxDensity::Function
    logApproxMarginals::Vector{Function}
    refiningFunctions::Vector{Function}

    function BaseModel(
        Hθ₀::DenseVector,
        logTargetDensity::Function,
        logApproxMarginals::Vector{Function},
        refiningFunctions::Vector{Function},
    )
        
        nHyperParams = length(Hθ₀);
        Hθ = Hθ₀[:, :];
        logApproxDensity(θ::DenseVector) = sum(f(θ, Hθ₀) for f in logApproxMarginals);
        new(
            nHyperParams,
            Hθ,
            logTargetDensity,
            logApproxDensity,
            logApproxMarginals,
            refiningFunctions,
        )
    end

end


"""
    getHθ(model)

Give the last upadated hyper-parameters.
"""
function getHθ(model::BaseModel)
    return model.Hθ[:, end]
end


end