using Test

include("../precipMeanField/CAVI.jl");

@testset "CAVI.jl" begin

    @testset "updateHyperParams!(Hθ; F, Y)" begin
        
        η = [float(i) for i = 1:9];
        bᵤ = 2;
        Hθ = [η..., bᵤ];
        F = iGMRF(3, 3, 1);
        Y = [float.([i*2, i*2+1]) for i = 0:8];

        updateHyperParams!(Hθ, F=F, Y=Y)

        @test Hθ[1] ≈ 16/7;
        @test Hθ[2] ≈ 430/133;

    end

end