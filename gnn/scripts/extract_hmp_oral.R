#!/usr/bin/env Rscript
# HMP 口腔 16S データ抽出 (Phase 2)
# Usage: Rscript extract_hmp_oral.R [output_dir]
# Requires: BiocManager::install("HMP16SData")

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "data/hmp_oral"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("HMP16SData", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install("HMP16SData")
}
library(HMP16SData)

message("Loading V35 (4,743 samples)...")
d <- V35()
cd <- as.data.frame(SummarizedExperiment::colData(d))
message("Body subsites: ", paste(unique(cd$HMP_BODY_SUBSITE)[1:5], collapse=", "), "...")

oral_keywords <- c("oral", "tongue", "throat", "saliva", "supragingival", "subgingival")
oral_idx <- grep(paste(oral_keywords, collapse = "|"), cd$HMP_BODY_SUBSITE, ignore.case = TRUE)
d_oral <- d[, oral_idx]
message("Oral samples: ", ncol(d_oral))

# Abundance matrix (samples x OTUs)
abund <- t(as.matrix(SummarizedExperiment::assay(d_oral)))
rd <- as.data.frame(SummarizedExperiment::rowData(d_oral))

# 5 菌種へのマッピング (Genus_Species で部分一致)
species_target <- c(
  "Streptococcus_oralis", "Actinomyces_naeslundii", "Veillonella_dispar",
  "Fusobacterium_nucleatum", "Porphyromonas_gingivalis"
)
lineage <- rd$CONSENSUS_LINEAGE
otu_to_species <- rep(NA, nrow(rd))
for (i in seq_along(species_target)) {
  pat <- gsub("_", " ", species_target[i])
  idx <- grep(pat, lineage, ignore.case = TRUE)
  otu_to_species[idx] <- species_target[i]
}

# 菌種別に OTU を集約
species_abund <- matrix(0, nrow = nrow(abund), ncol = 5)
colnames(species_abund) <- c("S_oralis", "A_naeslundii", "V_dispar", "F_nucleatum", "P_gingivalis")
for (j in 1:5) {
  idx <- which(otu_to_species == species_target[j])
  if (length(idx) > 0) {
    species_abund[, j] <- rowSums(abund[, idx, drop = FALSE])
  }
}
# 正規化 (volume fraction)
species_abund <- species_abund / rowSums(species_abund)
species_abund[is.nan(species_abund)] <- 0

out_abund <- file.path(out_dir, "species_abundance.csv")
write.csv(species_abund, out_abund, row.names = TRUE)
message("Saved: ", out_abund)

out_meta <- file.path(out_dir, "sample_metadata.csv")
write.csv(cd[oral_idx, ], out_meta, row.names = TRUE)
message("Saved: ", out_meta)
message("Done.")
