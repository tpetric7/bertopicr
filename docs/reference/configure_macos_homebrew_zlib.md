# Configure Homebrew zlib on macOS

Sets DYLD_FALLBACK_LIBRARY_PATH to Homebrew's zlib lib directory. This
can help reticulate find compatible libraries on macOS.

## Usage

``` r
configure_macos_homebrew_zlib(quiet = FALSE)
```

## Arguments

- quiet:

  Logical. If TRUE, suppresses messages.

## Value

Logical. TRUE if the environment was updated, FALSE otherwise.
