using Test, Random

include("../../src/instances/instanceNMF.jl");


@testset "instanceNMF.jl" begin
    
    @testset "InstanceNMF" begin
        
        m₁ = 3;
        m₂ = 3;

        instance = InstanceNMF(
            m₁,
            m₂,
            seed=400,
            biasmu=10.0,
            realkappa=10.0,
        )

        @test mean(instance.data[1]) - 10.0 < .1;
        @test var(instance.data[1]) - 1.0 < .1;
        @test instance.model.nHyperParams == length(instance.model.refiningFunctions);

    end

end