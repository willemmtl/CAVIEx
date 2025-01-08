using Test, Distributions, GMRF, OrderedCollections

if !isdefined(Main, :NormalMeanField)
    include("../src/models/NormalMeanField.jl");
end
using .NormalMeanField

include("ressources/CAVI.jl");


@testset "CAVI.jl" begin
    
    @testset "MonteCarloKL(model)" begin
        
        trueResult = .5 * (log(σ²) + 1/σ² - 1);
        approxError = abs(MonteCarloKL(modelMC) - trueResult);
        
        @test approxError < 0.02;

    end


    @testset "refineHyperParams!(model)" begin
        
        refineHyperParams!(refiningModel)

        @test refiningModel.hyperParams[:η1].trace[2] ≈ 1/4;
        @test refiningModel.hyperParams[:η2].trace[2] ≈ 21/16;
        @test refiningModel.hyperParams[:η3].trace[2] ≈ 37/16;
        @test refiningModel.hyperParams[:η4].trace[2] ≈ 133/32;
        @test refiningModel.hyperParams[:s²1].trace[2] ≈ 1/4;
        @test refiningModel.hyperParams[:s²2].trace[2] ≈ 1/4;
        @test refiningModel.hyperParams[:s²3].trace[2] ≈ 1/4;
        @test refiningModel.hyperParams[:s²4].trace[2] ≈ 1/4;
        @test refiningModel.hyperParams[:aᵤ].trace[2] ≈ 2.5;
        @test refiningModel.hyperParams[:bᵤ].trace[2] ≈ 8637/1024 + 1/100;

    end

end