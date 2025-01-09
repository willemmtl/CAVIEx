"""
Instance of the ImproperImproperNormalMeanField Model.

The data Xk1, ..., Xkn are drawn from Normal(μk, 1, 0).
The location parameters μk are drawn from iGMRF(κᵤ).
The target density is the posterior of θ = [μ..., κᵤ].
The mean-field aproximation gives
    μk ∼ Normal(ηk, s²k)
    κᵤ ∼ Gamma(aᵤ, bᵤ)
where [η..., s²..., aᵤ, bᵤ] are the hyper-parameters.
"""

using GMRF, OrderedCollections

if !isdefined(Main, :CAVIEx)
    include("../CAVIEx.jl");
end
using .CAVIEx

if !isdefined(Main, :ImproperNormalMeanField)
    include("../models/ImproperNormalMeanField.jl");
end
using .ImproperNormalMeanField

include("../dataGen/dataGenNormal.jl");


"""
    InstanceNMF

Generate an instance of a ImproperNormalMeanField model.

# Attributes :
- `F::iGMRF`: Spatial scheme.
- `gridTarget::Array{Float64, 3}`: True values of μ.
- `data::Vector{Vector{<:Real}}`: Extreme values for each cell.
- `model::CAVIEx.Model`: Normal Mean-Field configurations.
"""
struct InstanceINMF
    F::iGMRF
    gridTarget::Array{Float64, 3}
    data::Vector{Vector{<:Real}}
    model::CAVIEx.Model

    """
    Constructor.

    # Arguments
    - `m₁::Integer`: Number of rows of the grid.
    - `m₂::Integer`: Number of columns of the grid.
    - `seed::Integer`: Seed for data and true parameters generation.
    - `biasmu::Float64`: Value around which the true mean parameters are generated.
    - `realkappa::Float64`: True precision parameter of the iGMRF.
    """
    function InstanceINMF(
        m₁::Integer,
        m₂::Integer;
        seed::Integer,
        biasmu::Float64,
        realkappa::Float64,
    )
        Random.seed!(seed);
        F = iGMRF(m₁, m₂, 1, realkappa);
        gridTarget = generateTargetGrid(F);
        gridTarget[:, :, 1] = gridTarget[:, :, 1] .+ biasmu;
        nobs = 100;
        data = generateData(gridTarget, nobs);

        m = m₁ * m₂;
        params = OrderedDict(
            (
                Symbol("μ$k") => (
                    Normal,
                    OrderedDict(
                        :μ => (k, Symbol("η$k"), Hθ -> ImproperNormalMeanField.refine_η(Hθ, k, F=F, Y=data)),
                        :σ => (m+k, Symbol("s²$k"), Hθ -> ImproperNormalMeanField.refine_s²(Hθ, k, F=F, Y=data)),
                    ),
                )
                for k=1:m
            )...,
            :κᵤ => (
                Gamma,
                OrderedDict(
                    :α => (2*m+1, :aᵤ, Hθ -> ImproperNormalMeanField.refine_aᵤ(F=F)),
                    :θ => (2*m+2, :bᵤ, Hθ -> ImproperNormalMeanField.refine_bᵤ(Hθ, F=F)),
                )
            ),
        )

        new(
            F,
            gridTarget,
            data,
            CAVIEx.Model(
                params,
                θ -> ImproperNormalMeanField.logTargetDensity(θ, F=F, Y=data),
                (θ, Hθ) -> ImproperNormalMeanField.logApproxDensity(θ, Hθ, F=F),
                niter -> ImproperNormalMeanField.gibbs(niter, data, m₁=m₁, m₂=m₂, δ²=0.2, κᵤ₀=1, μ₀=zeros(m)),
            )
        )
    end

end