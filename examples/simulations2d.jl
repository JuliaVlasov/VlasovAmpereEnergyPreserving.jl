using Plots
using TimerOutputs

include("vlasov-maxwell-hongtao-lapenta.jl")
include("vlasov-maxwell-lie-1.jl")
include("vlasov-maxwell-lie-2.jl")

tmax = 10.0
lie1 = vm_lie_1(tmax)
lie2 = vm_lie_2(tmax)
lapenta = lapenta_2d(tmax)
plot(lapenta[!, "time"], lapenta[!, "ExNorm"]; yscale = :log10, label = "lapenta")
plot!(lie1[!, "time"], lie1[!, "ExNorm"]; yscale = :log10, label = "lie1")
plot!(lie2[!, "time"], lie2[!, "ExNorm"]; yscale = :log10, label = "lie2")
title!("Integral Ex norm")
png("simulations2d")
