using Distributions, Optim, ForwardDiff

# """
#     dataLevelLogLike(Y, θ)

# Log-likelihood at the data level for our given hierarchical model.

# Will be useful to compute the location parameters' Fisher Information.

# # Arguments :
# - `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
# - `μ::DenseVector`: Location parameters.
# """
# function dataLevelLogLike(Y::Vector{Vector{Float64}}, μ::DenseVector)
#     return sum(loglikelihood.(Gumbel.(μ, 1), Y))
# end


"""
    logFunctionalFormPosterior(θ; F, Y)

Evaluate the log functional form of the posterior distribution of our hierarchical model.

# Arguments :
- `θ::DenseVector`: Parameters of interest -> [μ..., κᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function logFunctionalFormPosterior(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.m₁ * F.G.m₂;

    return (
        sum(loglikelihood.(Gumbel.(θ[1:m], 1), Y)) 
        + (m - F.r) * log(θ[m+1]) / 2 
        - θ[m+1] * θ[1:m]' * F.G.W * θ[1:m] / 2
        - θ[m+1] / 100
    )

end;


"""
    logDensityApprox(x; η, s², aᵤ, bᵤ)

Evaluate the log-density of approximation of the variational inference.

# Arguments :
- `x::DenseVector`: Parameters of interest -> [μ..., κᵤ].
- `η::DenseVector`: Mean parameters of the location parameters' approximation.
- `s²::DenseVector`: Variance parameters of the location parameters' approximation.
- `aᵤ::Real`: First parameter of the precision's approximation.
- `bᵤ::Real`: Second parameter of the precision's approximation.
"""
function logDensityApprox(x::DenseVector; η::DenseVector, s²::DenseVector, aᵤ::Real, bᵤ::Real)
    return sum(logpdf.(Normal.(η, sqrt.(s²)), x[1:end-1])) + logpdf(Gamma(aᵤ, bᵤ), x[end]);
end;


"""
    findMode(f, θ₀)

Given a density's functional form, find a mode.

Warning : there may be several modes.
This function will catch the first one it finds depending on the initialization.

# Arguments :
- `f::Function`: A density's functional form (can be log).
- `θ₀::DenseVector`: Parameters' initialization.
"""
function findMode(f::Function, θ₀::DenseVector)
    F(θ::DenseVector) = -f(θ)
    return optimize(F, θ₀, Newton(); autodiff = :forward)
end;


"""
    computeFisherVariance(logf, θ̂)

Compute the quadratic approximation variance matrix of a given distribution.
It is based on the Fisher information.

The link is Σ = F⁻¹ where F is the Fisher information matrix.

# Arguments :
- `logf::Function`: log the density's functional form.
- `θ̂::AbstractArray`: mode of the distribution.
"""
function computeFisherVariance(logf::Function, θ̂::AbstractArray)
    return inv(computeFisherInformation(θ -> logf(θ), θ̂))
end


"""
    computeFisherInformation(logf, θ̂)

Compute the Fisher information matrix of a given distribution.

# Arguments :
- `logf::Function`: log the density's functional form.
- `θ̂::AbstractArray`: mode of the distribution.
"""
function computeFisherInformation(logf::Function, θ̂::AbstractArray)
    return - ForwardDiff.hessian(logf, θ̂)
end;