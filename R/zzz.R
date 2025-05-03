## File: R/zzz.R
.onLoad <- function(libname, pkgname) {

  ## macOS only ───────────────────────────────────────────────────────────
  if (identical(Sys.info()[["sysname"]], "Darwin")) {

    ## Try Home‑brew first
    brew_zlib <- try(
      system2("brew", c("--prefix", "zlib"), stdout = TRUE),
      silent = TRUE
    )

    if (!inherits(brew_zlib, "try-error") && dir.exists(brew_zlib)) {
      path <- file.path(brew_zlib, "lib")
      ## Use FALLBACK so it is honoured by SIP‑protected GUI apps
      Sys.setenv(DYLD_FALLBACK_LIBRARY_PATH = path)
    }
  }

  ## If you need to register S3 methods, Python modules, etc.
  # reticulate::configure_environment(pkgname)
}
