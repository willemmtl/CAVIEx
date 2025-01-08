"""
Specific functions for the PrecipMeanField Model.

The observations Xk1, ..., Xkn are drawn from GEV(μk, 1, 0).
The location parameters μk are drawn from iGMRF(κᵤ).
The target density is the posterior of θ = [μ..., κᵤ].
The mean-field aproximation gives
    μk ∼ Normal(ηk, s²k)...
    κᵤ ∼ Gamma(aᵤ, bᵤ)
where [..., ηk, s²k, ..., aᵤ, bᵤ] are the hyper-parameters.
"""

module PrecipMeanField

using Distributions, GMRF, LinearAlgebra
include("../mcmc/precipMCMC.jl");

"""
    logTargetDensity(θ; F, Y)

Evaluate the log functional form of the posterior distribution of our hierarchical model.

# Arguments :
- `θ::DenseVector`: Parameters of interest -> [μ..., κᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme observations for each cell.
"""
function logTargetDensity(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.gridSize[1] * F.G.gridSize[2];

    return (
        sum(loglikelihood.(Gumbel.(θ[1:m], 1), Y)) 
        + (m - F.rankDeficiency) * log(θ[m+1]) / 2 
        - θ[m+1] * θ[1:m]' * F.G.W * θ[1:m] / 2
        - θ[m+1] / 100
    )

end


"""
    logApproxDensity(θ, Hθ; F)

Evaluate the joint approximating density of the parameters.

It corresponds to the sum of all marginals (mean-field approximation).

# Arguments :
- `θ::DenseVector`: Parameters -> [μ₁, ..., μₘ, κᵤ].``
- `Hθ::DenseVector`: Hyper-parameters -> [η1, s²1, ..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
"""
function logApproxDensity(θ::DenseVector, Hθ::DenseVector; F::iGMRF)

    m = F.G.gridSize[1] * F.G.gridSize[2];

    return logpdf(Gamma(Hθ[2*m+1], 1/Hθ[2*m+2]), θ[end]) + sum([logpdf(Normal(Hθ[k], sqrt(Hθ[m+k])), θ[k]) for k = 1:m])
end


"""
    refine_η(Hθ, k; F, Y)

Refine the mean parameter for location parameter's approximation.

See the mathematical formula in Notion.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters [η..., s..., aᵤ, bᵤ].
- `k::Integer`: Numero of the cell to update.
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme observations for each cell.
"""
function refine_η(Hθ::DenseVector, k::Integer; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.gridSize[1] * F.G.gridSize[2];
    η = Hθ[1:m];
    aᵤ = Hθ[2*m+1];
    bᵤ = Hθ[2*m+2];

    Sη = (- F.G.W̄ * η)[k]
    Sy = sum(Y[k]);
    n_obs = length(Y[k]);
    n_neighbors = F.G.W[k, k];
    
    return (aᵤ * Sη + bᵤ * Sy) / (bᵤ * n_obs + aᵤ * n_neighbors)
end


"""
    refine_s²(Hθ, k; F, Y)

Refine the approximation variances for the location parameters.
    
See the mathematical formula in Notion.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s..., aᵤ, bᵤ].
- `k::Integer`: Numero of the cell to update.
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme observations for each cell.
"""
function refine_s²(Hθ::DenseVector, k::Integer; F::iGMRF, Y::Vector{Vector{Float64}})
        
    m = F.G.gridSize[1] * F.G.gridSize[2];
    n_obs = length(Y[k]);
    n_neighbors = F.G.W[k, k];
    aᵤ = Hθ[2*m+1];
    bᵤ = Hθ[2*m+2];
    
    return bᵤ / (bᵤ * n_obs + aᵤ * n_neighbors)
end


"""
    refine_aᵤ(; F)

Refine the first parameter for precision's approximation.

See the mathematical formula in Notion.

# Arguments :
- `F::iGMRF`: Spatial scheme.
"""
function refine_aᵤ(; F::iGMRF)

    m = F.G.gridSize[1] * F.G.gridSize[2];

    return (m-1)/2 + 1
end


"""
    refine_bᵤ(Hθ; F)

Refine the second parameter for precision's approximation.

See the mathematical formula in Notion.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
"""
function refine_bᵤ(Hθ::DenseVector; F::iGMRF)
    
    m = F.G.gridSize[1] * F.G.gridSize[2];
    η = Hθ[1:m];
    
    return 0.5 * η' * F.G.W * η + 0.01
end


# """
#     initializeHyperParams(F, Y)

# Initialize hyperparameters.

# η is initialized with a quadratic approximation of the posteriori.
# b can be computed thanks to η.

# # Arguments :
# - `F::iGMRF`: Spatial scheme.
# - `Y::Vector{Vector{Float64}}`: Extreme observations for each cell.
# """
# function initializeHyperParams(F::iGMRF, Y::Vector{Vector{Float64}})

#     m = F.G.gridSize[1] * F.G.gridSize[2];
    
#     mode = findMode(θ -> logFunctionalFormPosterior(θ, F=F, Y=Y), [fill(0.0, m)..., 1]);
#     α = Optim.minimizer(mode);

#     Fvar = computeFisherVariance(θ -> logFunctionalFormPosterior(θ, F=F, Y=Y), α);

#     α₀ = α[2:end]
#     S₀ = round.(Fvar[2:end, 2:end] , digits=5);

#     η₀ = Distributions.rand(MvNormal(α₀, S₀));
#     b₀ = (η₀' * F.G.W * η₀) / 2 + 1 / 100;

#     return [η₀..., b₀];

# end;


end