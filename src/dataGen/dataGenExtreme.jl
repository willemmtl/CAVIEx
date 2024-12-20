using Random, Distributions, SparseArrays, GMRF

function generateData(grid_params::Array{<:Real}, nobs::Integer)

    # Vecteur qui contiendra les observations pour chaque cellule
    Y = Vector{Float64}[]

    for i = 1:size(grid_params, 1)
        for j = 1:size(grid_params, 2)
            # Récupération des paramètres de la cellule courante
            gev_params = grid_params[i, j, :]
            # Génération des observations pour la cellule courante
            y = rand(GeneralizedExtremeValue(gev_params...), nobs)
            push!(Y, y)
        end
    end

    return Y
end

function generateTargetGrid(F::iGMRF)
    # Paramètre de position
    μ = generateParams(F)
    # Paramètre d'échelle
    ϕ = zeros(F.G.gridSize...)
    # Paramètre de forme
    ξ = zeros(F.G.gridSize...)
    # Concatène les paramètres pour former la grille finale m₁xm₂x3
    return cat(μ, exp.(ϕ), ξ, dims=3)
end

function generateParams(F::iGMRF)
    # Génère les effets spatiaux
    s = rand(F)
    # Il n'y a pas de variable explicative
    # On renvoie donc directement les effets spatiaux
    return reshape(s, F.G.gridSize...)'
end