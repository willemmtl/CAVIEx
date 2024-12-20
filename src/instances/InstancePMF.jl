using GMRF

if !isdefined(Main, :AbstractModel)
    include("../models/AbstractModel.jl");
end
using .AbstractModel
if !isdefined(Main, :PrecipMeanField)
    include("../models/PrecipMeanField.jl");
end
using .PrecipMeanField

include("../dataGen.jl");


"""
    InstancePMF

Generate an instance of a PrecipMeanField model.

# Attributes :
- `F::iGMRF`: Spatial scheme.
- `gridTarget::Array{Float64, 3}`: True values of μ.
- `data::Vector{Vector{<:Real}}`: Extreme values for each cell.
- `model::AbstractModel.BaseModel`: Every function needed in the model.
    -> See BaseModel.
"""
struct InstancePMF
    F::iGMRF
    gridTarget::Array{Float64, 3}
    data::Vector{Vector{<:Real}}
    model::AbstractModel.BaseModel

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
        hyperParams = [["η$k" for k = 1:m]..., ["s²$k" for k = 1:m]..., "aᵤ", "bᵤ"];

        new(
            F,
            gridTarget,
            data,
            AbstractModel.BaseModel(
                hyperParams,
                θ -> PrecipMeanField.logTargetDensity(θ, F=F, Y=data),
                (θ, Hθ) -> PrecipMeanField.logApproxDensity(θ, Hθ, F=F),
                [
                    [Hθ -> PrecipMeanField.μMarginal(Hθ, k=k, F=F) for k = 1:4]...,
                    Hθ -> PrecipMeanField.κMarginal(Hθ, F=F)
                ],
                [
                    [Hθ -> PrecipMeanField.refine_η(Hθ, k, F=F, Y=data) for k = 1:4]...,
                    [Hθ -> PrecipMeanField.refine_s²(Hθ, k, F=F, Y=data) for k = 1:4]...,
                    Hθ -> PrecipMeanField.refine_aᵤ(F=F),
                    Hθ -> PrecipMeanField.refine_bᵤ(Hθ, F=F),
                ]
            )
        )
    end

end