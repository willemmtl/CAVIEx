using Test

include("../../src/instances/InstanceDemo.jl");

@testset "instanceDemo.jl" begin
    
    @testset "InstanceDemo" begin
        
        realmu = 75.0;
        realsigma2 = 100.0;

        instance = InstanceDemo(
            seed=400,
            realmu=realmu,
            realsigma2=realsigma2,
        )

        @test mean(instance.data) - 75.0 < .1;
        @test var(instance.data) - 100.0 < .1;
        @test instance.model.hyperParams[:m].numero == 1;

    end

end