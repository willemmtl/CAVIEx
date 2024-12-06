using Optim, Distributions, Gadfly

include("../precipModel/iGMRF.jl");
include("../precipModel/framework.jl");


"""
    runCAVI(n_epoch, epoch_size; F, Y)

Run the CAVI algorithm over the given data and spatial scheme.

# Arguments :
- `n_epoch::Integer`: Number of epochs to compute i.e. number of type we'll compute the convergence criterion.
- `epoch_size::Integer`: Size of each epoch i.e. number of iterations before computing the convergence criterion.
- `F::iGMRF`: Spatial scheme containing the structure matrix and rank defficiency.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function runCAVI(n_epoch::Integer, epoch_size::Integer; F::iGMRF, Y::Vector{Vector{Float64}})

    ### ---- Initialization ---- ###
    duration = @elapsed begin
        Hθ = initializeHyperParams(F, Y);
    end
    println("Initialization made in ", duration, " s")
    
    ### ---- Ascent ---- ###
    MCKL = Float64[];
    duration = @elapsed begin
        for _ = 1:n_epoch
            push!(MCKL, MonteCarloKL(Hθ, F=F, Y=Y))
            for _ = 1:epoch_size
                updateHyperParams!(Hθ, F=F, Y=Y);
            end
        end
    end
    println("Minimization made in ", duration, " s")
    
    return MCKL, Hθ
end;


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


"""
    MonteCarloKL(Hθ; F, Y)

Compute the convergence criterion with current hyper-parameters.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters [η..., bᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`:: Extreme data for each cell.
"""
function MonteCarloKL(Hθ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.m₁ * F.G.m₂;
    
    N = 1000;
    Θ = zeros(m+1, N);

    # Unpacking
    η = Hθ[1:m];
    bᵤ = Hθ[m + 1];
    aᵤ = compute_aᵤ(m);
    s² = compute_s²(aᵤ, bᵤ, length(Y[1]), [diag(F.G.W)...]);
    
    for i = 1:m
        Θ[i, :] = rand(Normal(η[i], sqrt(s²[i])), N);
    end
    Θ[m+1, :] = rand(Gamma(aᵤ, bᵤ), N);
    
    logTarget = evaluateLogMvDensity(x -> logFunctionalFormPosterior(x; F=F, Y=Y), Θ);
    logApprox = evaluateLogMvDensity(x -> logDensityApprox(x, η=η, s²=s², aᵤ=aᵤ, bᵤ=bᵤ), Θ);
    
    return sum(logApprox .- logTarget) / N
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
    updateHyperParams!(Hθ; F, Y)

Update approximation parameters η and bᵤ.

Correspond to one iteration in an epoch of the CAVI algorithm.

# Arguments :
- `Hθ::DenseVector`: Last updated hyper-parameters [η..., bᵤ].
- `F::iGMRF`: Spatial scheme.
- `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
"""
function updateHyperParams!(Hθ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    
    m = F.G.m₁ * F.G.m₂;
    aᵤ = compute_aᵤ(m);

    # Update η
    for k = 1:m
        Hθ[k] = compute_η(k, Hθ[1:m], aᵤ=aᵤ, bᵤ=Hθ[m+1], F=F, Y=Y)
    end
    # Update b
    Hθ[m+1] = compute_bᵤ(Hθ[1:m], F);
    
end;


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