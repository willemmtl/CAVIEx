using Test, Distributions

@testset "Param.jl" begin

    param = CAVIEx.Param(Normal);
    
    @testset "Param(approxDistribution)" begin
        
        @test param.approxDistribution == Normal;
        @test isnothing(param.approxMarginal);

    end


    @testset "draw(param, N)" begin
        
        param.approxMarginal = Normal(100.0, 1.0);
        N = 100;
        sample = CAVIEx.draw(param, N);

        @test length(sample) == N;

    end

end