module CAVIEx

export 
    runCAVI,
    MonteCarloKL,
    refineHyperParams!,
    initialize!,
    syncMarginals!,
    getHpValues,
    generateApproxSample,
    runMCMC!,
    evaluateLogMvDensity,
    adaptHp!,
    plotConvergenceCriterion,
    plotTraceCAVI,
    plotApproxVSMCMC
    plotTraceMCMC,

include("CAVI.jl");
include("plotting.jl");

end