using Test, Random, Statistics

include("../../src/instances/instancePMF.jl");


@testset "instancePMF.jl" begin
    
    @testset "InstancePMF" begin
        
        m₁ = 3;
        m₂ = 3;

        instance = InstancePMF(
            m₁,
            m₂,
            seed=400,
            biasmu=10.0,
            realkappa=10.0,
        )

        @test instance.data[1][1] ≈ 12.215413935213412;
        @test instance.gridTarget[1, 1, 2] ≈ 1.0;
        @test instance.gridTarget[2, 2, 3] ≈ 0.0;
    end

end