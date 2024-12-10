using Distributions

include("../precipFramework/iGMRF.jl");


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
    return sum(logpdf.(Normal.(η, sqrt.(s²)), x[1:end-1])) + logpdf(Gamma(aᵤ, 1/bᵤ), x[end]);
end;


"""
    compute_s²(aᵤ, bᵤ, n, n_neighbors)

Compute the approximation variances for the location parameters.

See the mathematical formula in Notion.

# Arguments :
- `aᵤ::Real`: First parameter of the approximation Gamma density for the precision parameter.
- `bᵤ::Real`: Second parameter of the approximation Gamma density for the precision parameter.
- `n_obs::Integer`: Number of observations per cell (the same for all cells for now).
- `n_neighbors::Vector{<:Real}`: Number of neighbors for each cell.
"""
function compute_s²(aᵤ::Real, bᵤ::Real, n_obs::Integer, n_neighbors::Vector{<:Real})
    return bᵤ ./ (bᵤ .* n_obs .+ aᵤ .* n_neighbors)
end


"""
    compute_aᵤ(m)

Compute the first parameter for precision's approximation.

See the mathematical formula in Notion.

# Arguments :
- `m::Integer`: Number of cells.
"""
function compute_aᵤ(m::Integer)
    return (m-1)/2 + 1
end


"""
    compute_η(k, η; aᵤ, bᵤ, F, Y)

Compute the mean parameter for location parameter's approximation.

See the mathematical formula in Notion.

# Arguments :
- `k::Integer`: Numero of the cell to update.
- `η::DenseVector`: Current values of the mean parameters for location parameter's approximation.
- `aᵤ::Real`: First parameter for precision's approximation.
- `bᵤ::Real`: Second parameter for precision's approximation.
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function compute_η(k::Integer, η::DenseVector; aᵤ::Real, bᵤ::Real, F::iGMRF, Y::Vector{Vector{Float64}})
    
    Sη = (- F.G.W̄ * η)[k]
    Sy = sum(Y[k]);
    n_obs = length(Y[k]);
    n_neighbors = F.G.W[k, k];

    return (aᵤ * Sη + bᵤ * Sy) / (bᵤ * n_obs + aᵤ * n_neighbors)
end


"""
    compute_bᵤ(η, F)

Compute the second parameter for precision's approximation.

See the mathematical formula in Notion.

# Arguments :
- `η::DenseVector`: Mean approximation vector for location parameters.
- `F::iGMRF`: Spatial scheme.
"""
function compute_bᵤ(η::DenseVector, F::iGMRF)
    return 0.5 * η' * F.G.W * η + 0.01
end