module NormalMeanField

using Distributions, GMRF


"""
    logTargetDensity(θ; F, Y)

Evaluate the log functional form of the posterior distribution of our hierarchical model.

# Arguments :
- `θ::DenseVector`: Parameters of interest -> [μ..., κᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Vector of observations for each cell.
"""
function logTargetDensity(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.gridSize[1] * F.G.gridSize[2];

    return (
        sum(loglikelihood.(Normal.(θ[1:m], 1), Y)) 
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
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
"""
function logApproxDensity(θ::DenseVector, Hθ::DenseVector; F::iGMRF)

    m = F.G.gridSize[1] * F.G.gridSize[2];

    return logpdf(κMarginal(Hθ, F=F), θ[end]) + sum([logpdf(μMarginal(Hθ, k=k, F=F), θ[k]) for k = 1:m])
end


"""
    μMarginal(Hθ; k, F)

Marginal approximating distribution of μₖ.

It's a Normal distribution with mean η and variance s².

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `k::Integer`: Numero of the cell.
- `F::iGMRF`: Spatial scheme.
"""
function μMarginal(Hθ::DenseVector; k::Integer, F::iGMRF)
    
    m = F.G.gridSize[1] * F.G.gridSize[2];
    η = Hθ[1:m];
    s² = Hθ[m+1:2*m];
    
    return Normal(η[k], sqrt.(s²[k]))
end


"""
    κMarginal(Hθ; F)

Evaluate the marginal approximating density of κᵤ.

It's a Gamma distribution with parameters aᵤ and bᵤ.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
"""
function κMarginal(Hθ::DenseVector; F::iGMRF)
    
    m = F.G.gridSize[1] * F.G.gridSize[2];
    aᵤ = Hθ[2*m+1];
    bᵤ = Hθ[2*m+2];

    return Gamma(aᵤ, 1/bᵤ)
end


"""
    refine_η(Hθ, k; F, Y)

Refine the mean parameter for location parameter's approximation.

See the mathematical formula in Notion.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters [η..., s²..., aᵤ, bᵤ].
- `k::Integer`: Numero of the cell to update.
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
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
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `k::Integer`: Numero of the cell to update.
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
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
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
"""
function refine_bᵤ(Hθ::DenseVector; F::iGMRF)
    
    m = F.G.gridSize[1] * F.G.gridSize[2];
    η = Hθ[1:m];
    
    return 0.5 * η' * F.G.W * η + 0.01
end

end