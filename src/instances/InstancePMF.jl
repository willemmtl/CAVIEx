using GMRF, OrderedCollections

if !isdefined(Main, :CAVIEx)
    include("../CAVIEx.jl");
end
using .CAVIEx

if !isdefined(Main, :PrecipMeanField)
    include("../models/PrecipMeanField.jl");
end
using .PrecipMeanField

include("../dataGen/dataGenExtreme.jl");


"""
    InstancePMF

Generate an instance of a PrecipMeanField model.

# Attributes :
- `F::iGMRF`: Spatial scheme.
- `gridTarget::Array{Float64, 3}`: True values of μ.
- `data::Vector{Vector{<:Real}}`: Extreme values for each cell.
- `model::CAVIEx.Model`: TBD.
"""
struct InstancePMF
    F::iGMRF
    gridTarget::Array{Float64, 3}
    data::Vector{Vector{<:Real}}
    model::CAVIEx.Model

    function InstancePMF(
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
                        :μ => (k, Symbol("η$k"), Hθ -> PrecipMeanField.refine_η(Hθ, k, F=F, Y=data)),
                        :σ => (m+k, Symbol("s²$k"), Hθ -> PrecipMeanField.refine_s²(Hθ, k, F=F, Y=data)),
                    )
                )
                for k = 1:m
            )...,
            :κᵤ => (
                Gamma,
                OrderedDict(
                    :α => (2*m+1, :aᵤ, Hθ -> PrecipMeanField.refine_aᵤ(F=F)),
                    :θ => (2*m+2, :bᵤ, Hθ -> PrecipMeanField.refine_bᵤ(Hθ, F=F)),
                ),
            )
        );

        new(
            F,
            gridTarget,
            data,
            CAVIEx.Model(
                params,
                θ -> PrecipMeanField.logTargetDensity(θ, F=F, Y=data),
                (θ, Hθ) -> PrecipMeanField.logApproxDensity(θ, Hθ, F=F),
                niter -> PrecipMeanField.gibbs(niter, data, m₁=m₁, m₂=m₂, δ²=0.2, κᵤ₀=1, μ₀=zeros(m)),
            )
        )
    end

end