using Test, Random

include("../../src/instances/instanceDemo.jl");


@testset "instanceDemo.jl" begin
    
    @testset "InstanceDemo" begin
        
        realmu = 75.0;
        realsigma2 = 100.0;

        instance = InstanceDemo(
            seed=400,
            realmu=realmu,
            realsigma2=realsigma2,
        )

        @test mean(instance.data[1]) - 75.0 < .1;
        @test var(instance.data[1]) - 100.0 < .1;
        @test instance.model.nHyperParams == length(instance.model.refiningFunctions);

    end

end