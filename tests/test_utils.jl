using Test

@testset "utils.jl" begin
    
    @testset "evaluateLogMvDensity(f, supp)" begin

        supp = [1 2; 3 4];
        
        f(x::DenseVector) = x[1] + x[2]^2;

        res = evaluateLogMvDensity(f, supp);

        @test res[1] == 10;
        @test res[2] == 18;

    end
    
    
    @testset "adaptHp!(dist, values)" begin

        dist = Gamma;
        values = [1.0, 2.0];

        adaptHp!(Gamma, values)

        @test values[2] == .5;

    end

end