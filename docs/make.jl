using Documenter
using MultivariateFunctions

makedocs(
    format = :html,
    sitename = "MultivariateFunctions",
    modules = [MultivariateFunctions],
    doctest = true,
    root    = "<current-directory>",
    source  = "src",
    build   = "build",
    clean   = true,
)

deploydocs(
    repo   = "github.com/s-baumann/MultivariateFunctions.jl.git",
    julia  = "1.0",
    target = "build",
    deps   = nothing,
    make   = nothing
)
