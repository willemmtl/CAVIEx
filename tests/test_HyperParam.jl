using Test

@testset "HyperParam.jl" begin

    hyperParam = CAVIEx.HyperParam(x -> x + 1);
    
    @testset "HyperParam(refiningFunction)" begin
        
        @test hyperParam.refiningFunction(0) == 1;
        @test isnothing(hyperParam.trace);

    end

    @testset "current(hp)" begin
        
        @test_throws ErrorException CAVIEx.current(hyperParam);
        hyperParam.trace = [0.0];
        @test CAVIEx.current(hyperParam) == 0.0;

    end

end