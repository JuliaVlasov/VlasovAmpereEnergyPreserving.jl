using Documenter
using VlasovAmpereEnergyPreserving

makedocs(;
    sitename = "VlasovAmpereEnergyPreserving.jl",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Introduction" => "index.md",
        "Two-stream instability" => "two-stream.md",
        "Weibel instability" => "weibel.md",
        "Landau–Weibel instability" => "landau-weibel.md",
        "JAX acceleration benchmark" => "jax-benchmark.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaVlasov/VlasovAmpereEnergyPreserving.jl.git",
    devbranch = "main",
)
