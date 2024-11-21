
# NOTE: You may need to install.packages("[file names]") if you haven't already
# install.packages("spatstat")
install.packages("sjmisc")

# Matt: Function to install missing packages
install_if_missing <- function(packages) {
    # Loop through the list of packages
    for (pkg in packages) {
        # If the package is not installed, install it
        if (!require(pkg, character.only = TRUE)) {
            install.packages(pkg, dependencies = TRUE)
            library(pkg, character.only = TRUE)
        }
    }
}
packages <- c(
  "MASS", "jtools", "dplyr", "boot", "tidyverse", "texreg", "ggplot2", 
  "knitr", "lfe", "corrplot", "patchwork", "reshape", "styler", 
  "spatstat", "sjmisc", "stringr", "forcats", "latex2exp", 
  "figpatch", "khroma", "hrbrthemes", "gridExtra", "remotes"
)

install_if_missing(packages)


library(MASS)
library(jtools)
library(dplyr)
library(boot)
library(tidyverse)
library(texreg)
library(ggplot2)
library(knitr)
library(lfe)
library(corrplot)
library(patchwork)
library(reshape)
library(styler)
library(spatstat)
library(sjmisc)
library(stringr)
library(forcats)
library(latex2exp)
library(figpatch)
library(khroma)
library(hrbrthemes)
library(gridExtra)

# setwd('~/OneDrive - Northeastern University/neu/rezoning-schools/analysis/')
opts_chunk$set(tidy.opts=list(width.cutoff=80),tidy=TRUE)