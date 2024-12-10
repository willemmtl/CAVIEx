using Random, Distributions, SparseArrays

include("../precipFramework/iGMRF.jl");

function generateData(grid_params::Array{<:Real}, nobs::Integer)

    # Vecteur qui contiendra les observations pour chaque cellule
    Y = Vector{Float64}[]

    for i = 1:size(grid_params, 1)
        for j = 1:size(grid_params, 2)
            # Récupération des paramètres de la cellule courante
            params = grid_params[i, j, :]
            # Génération des observations pour la cellule courante
            y = rand(Normal(params...), nobs)
            push!(Y, y)
        end
    end

    return Y
end

function generateTargetGrid(F::iGMRF)
    # Moyenne
    μ = generateGEVParam(F)
    # Variance
    σ² = ones(F.G.m₁, F.G.m₂)
    # Concatène les paramètres pour former la grille finale m₁xm₂x3
    return cat(μ, σ², dims=3)
end

function generateGEVParam(F::iGMRF)
    # Nombre total de cellules
    m = F.G.m₁ * F.G.m₂
    # Génère les effets spatiaux
    s = sampleIGMRF(F)
    # Il n'y a pas de variable explicative
    # On renvoie donc directement les effets spatiaux
    return reshape(s, F.G.m₁, F.G.m₂)'
end