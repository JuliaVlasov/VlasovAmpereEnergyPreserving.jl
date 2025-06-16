include("vlasov-maxwell-hongtao-lapenta.jl")
include("vlasov-maxwell-lie-1.jl")
include("vlasov-maxwell-lie-2.jl")

tmax = 30.0
@time lapenta = lapenta_2d(tmax)
@time lie1 = vm_lie_1(tmax)
@time lie2 = vm_lie_2(tmax)
plot(lapenta[!, "time"], lapenta[!, "ExNorm"]; yscale=:log10, label = "lapenta")
plot!(lie1[!, "time"], lie1[!, "ExNorm"]; yscale=:log10, label="lie1")
plot!(lie2[!, "time"], lie2[!, "ExNorm"]; yscale=:log10, label="lie2")
title!("Integral Ex norm")
