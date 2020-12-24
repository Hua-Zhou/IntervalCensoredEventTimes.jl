using Documenter, IntervalCensoredEventTimes

makedocs(
    modules = [IntervalCensoredEventTimes],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Hua Zhou",
    sitename = "IntervalCensoredEventTimes.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Hua-Zhou/IntervalCensoredEventTimes.jl.git",
    push_preview = true
)
