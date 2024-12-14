module PrecipMeanField

using Distributions

include("../iGMRF.jl");


"""
    logTargetDensity(θ; F, Y)

Evaluate the log functional form of the posterior distribution of our hierarchical model.

# Arguments :
- `θ::DenseVector`: Parameters of interest -> [μ..., κᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function logTargetDensity(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.m₁ * F.G.m₂;

    return (
        sum(loglikelihood.(Gumbel.(θ[1:m], 1), Y)) 
        + (m - F.r) * log(θ[m+1]) / 2 
        - θ[m+1] * θ[1:m]' * F.G.W * θ[1:m] / 2
        - θ[m+1] / 100
    )

end


"""
    μMarginal(θ, Hθ; k, F)

Evaluate the marginal approximating density of μₖ.

It's a Normal distribution with mean η and variance s².

# Arguments :
- `θ::DenseVector`: Parameters -> [μ₁, ..., μₘ, κᵤ].``
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `k::Integer`: Numero of the cell.
- `F::iGMRF`: Spatial scheme.
"""
function μMarginal(θ::DenseVector, Hθ::DenseVector; k::Integer, F::iGMRF)
    
    m = F.G.m₁ * F.G.m₂;
    η = Hθ[1:m];
    s² = Hθ[m+1:2*m];
    
    return logpdf(Normal(η[k], sqrt.(s²[k])), θ[k])
end


"""
    κMarginal(θ, Hθ; F)

Evaluate the marginal approximating density of κᵤ.

It's a Gamma distribution with parameters aᵤ and bᵤ.

# Arguments :
- `θ::DenseVector`: Parameters -> [μ₁, ..., μₘ, κᵤ].
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
"""
function κMarginal(θ::DenseVector, Hθ::DenseVector; F::iGMRF)
    
    m = F.G.m₁ * F.G.m₂;
    aᵤ = Hθ[2*m+1];
    bᵤ = Hθ[2*m+2];

    return logpdf(Gamma(aᵤ, 1/bᵤ), θ[end])
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
    
    m = F.G.m₁ * F.G.m₂;
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
    refine_s²(Hθ; F, Y)

Refine the approximation variances for the location parameters.

See the mathematical formula in Notion.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [η..., s²..., aᵤ, bᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function refine_s²(Hθ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.m₁ * F.G.m₂;
    n_obs = length(Y[1]);
    n_neighbors = [diag(F.G.W)...];
    aᵤ = Hθ[2*m+1];
    bᵤ = Hθ[2*m+2];

    return bᵤ ./ (bᵤ .* n_obs .+ aᵤ .* n_neighbors)
end


"""
    refine_aᵤ(; F)

Refine the first parameter for precision's approximation.

See the mathematical formula in Notion.

# Arguments :
- `F::iGMRF`: Spatial scheme.
"""
function refine_aᵤ(; F::iGMRF)

    m = F.G.m₁ * F.G.m₂;

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
    
    m = F.G.m₁ * F.G.m₂;
    η = Hθ[1:m];
    
    return 0.5 * η' * F.G.W * η + 0.01
end


"""
    initializeHyperParams(F, Y)

Initialize hyperparameters.

η is initialized with a quadratic approximation of the posteriori.
b can be computed thanks to η.

# Arguments :
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function initializeHyperParams(F::iGMRF, Y::Vector{Vector{Float64}})

    m = F.G.m₁ * F.G.m₂;
    
    mode = findMode(θ -> logFunctionalFormPosterior(θ, F=F, Y=Y), [fill(0.0, m)..., 1]);
    α = Optim.minimizer(mode);

    Fvar = computeFisherVariance(θ -> logFunctionalFormPosterior(θ, F=F, Y=Y), α);

    α₀ = α[2:end]
    S₀ = round.(Fvar[2:end, 2:end] , digits=5);

    η₀ = rand(MvNormal(α₀, S₀));
    b₀ = (η₀' * F.G.W * η₀) / 2 + 1 / 100;

    return [η₀..., b₀];

end;


end