# Installs TCGAbiolinks / DESeq2 / SummarizedExperiment into the project R library.
# Target library = $R_LIBS_USER (set by run_route_b.ps1 / tools/slc_mars_env.ps1);
# falls back to ./outputs/R-libs when run standalone from the repo root.
user_lib <- Sys.getenv("R_LIBS_USER", "outputs/R-libs")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", lib = user_lib)
}
required_packages <- c("TCGAbiolinks", "DESeq2", "SummarizedExperiment")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  BiocManager::install(missing_packages, lib = user_lib, ask = FALSE, update = FALSE)
}
ok <- all(vapply(required_packages, requireNamespace, logical(1), quietly = TRUE))
cat("Required packages ready:", ok, "\n")
cat("Library paths:", paste(.libPaths(), collapse = " | "), "\n")
if (!ok) quit(status = 1)
