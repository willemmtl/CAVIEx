using Test, Distributions, GMRF

include("../src/models/AbstractModel.jl");
using .AbstractModel
include("../src/models/PrecipMeanField.jl");
using .PrecipMeanField

include("../src/CAVI.jl");

@testset "CAVI.jl" begin
    
    @testset "MonteCarloKL(model)" begin

        σ² = 100;
        hyperParams = ["θ"];

        N1 = Normal(0, 1);
        N2 = Normal(0, sqrt(σ²));

        model = AbstractModel.BaseModel(
            hyperParams,
            θ -> logpdf(N2, θ),
            (θ, Hθ) -> logpdf(N1, θ),
            [
                Hθ -> N1,
            ],
            [
                Hθ -> nothing,
            ]
        )

        model.hyperParamsValue[1] = 1.0;
        
        @test MonteCarloKL(model) - .5 * (log(σ²) + 1/σ² - 1) < 0.01;
    end


    @testset "refineHyperParams!(model)" begin
        
        m = 3 * 3;
        F = iGMRF(3, 3, 1, NaN);
        Y = [float.([i*2, i*2+1]) for i = 0:m-1];
        hyperParams = [["η$k" for k = 1:m]..., ["s²$k" for k = 1:m]..., "aᵤ", "bᵤ"];

        precipModel = AbstractModel.BaseModel(
            hyperParams,
            θ -> PrecipMeanField.logTargetDensity(θ, F=F, Y=Y),
            (θ, Hθ) -> PrecipMeanField.logApproxDensity(θ, Hθ, F=F),
            [
                [Hθ -> PrecipMeanField.μMarginal(Hθ, k=k, F=F) for k = 1:m]...,
                Hθ -> PrecipMeanField.κMarginal(Hθ, F=F)
            ],
            [
                [Hθ -> PrecipMeanField.refine_η(Hθ, k, F=F, Y=Y) for k = 1:m]...,
                [Hθ -> PrecipMeanField.refine_s²(Hθ, k, F=F, Y=Y) for k = 1:m]...,
                Hθ -> PrecipMeanField.refine_aᵤ(F=F),
                Hθ -> PrecipMeanField.refine_bᵤ(Hθ, F=F),
            ]
        )
        Hθ₀ = [[float(i) for i = 1:m]..., ones(m)..., 1.0, 2.0];
        precipModel.hyperParamsValue .= Hθ₀;

        refineHyperParams!(precipModel)

        @test precipModel.hyperParamsValue[1] ≈ 16/7;
        @test precipModel.hyperParamsValue[2] ≈ 430/133;
    end

end