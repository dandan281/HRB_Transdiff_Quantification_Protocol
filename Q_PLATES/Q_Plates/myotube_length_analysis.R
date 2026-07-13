# =====================================================================
# Myotube length analysis for Q_Plates
# For every quantified well: read the Length column, split myotubes at
# 300 um, and plot % below (black) vs % above (grey) per novelkin,
# combining replicates across plates with SEM error bars.
#
# R isn't installed on the machine this was authored on, so the figure
# was produced with the equivalent Python script; this R script
# reproduces the same analysis and chart. Requires: ggplot2.
# Run:  Rscript myotube_length_analysis.R
# =====================================================================

library(ggplot2)

BASE   <- "c:/Users/liqig/Desktop/Q_Plates"
THRESH <- 300          # micrometers

# --- One primary results file per well.
#     Combining rules: BMPR2 == BR223 (same mini binder); token order ignored;
#     m2 / 11m2 mutants merged into their base pair; replicates combined per group.
#     File choice per well: prefer plain _Results.csv, else _N, else _TBC.
files <- data.frame(
  plate = c("PLATE_23","PLATE_26","PLATE_28","PLATE_32",
            "PLATE_23","PLATE_28","PLATE_28",
            "PLATE_23","PLATE_26","PLATE_28","PLATE_32","PLATE_32",
            "PLATE_23","PLATE_32","PLATE_32",
            "PLATE_28","PLATE_32","PLATE_32",
            "PLATE_23","PLATE_32",
            "PLATE_23","PLATE_26","PLATE_32",
            "PLATE_32"),
  file  = c("B02_Ctrl_Results.csv","B02_Ctrl_Results.csv","B02_Ctrl_Results.csv","B02_Ctrl_Results.csv",
            "C05_BR223_EGFR_Results.csv","B04_BR223_EGFR_Results.csv","E08_BR223_EGFR_Results.csv",
            "C08_BR223_IGF1R_Results.csv","C08_BR223_IGF1R_Results.csv","E10_BR223_IGF1R_Results.csv",
              "D08_IGF1R_BR223_Results_TBC.csv","C11_IGF1R_BMPR2-11m2_Results.csv",
            "C09_BR223_TrkA_Results.csv","D09_TrkA_BR223_Results.csv","D02_TrkA_BMPR2-11m2_Results.csv",
            "B08_BMPR2_HER2_Results.csv","D11_HER2_BR223_Results_N.csv","D04_HER2_BR223-m2_Results.csv",
            "B03_ACT104_EGFR_Results.csv","C03_EGFR_ACT104_Results.csv",
            "B06_ACT104_TrkA_Results.csv","B06_ACT104_TrkA_Results.csv","C05_TrkA_ACT104_Results.csv",
            "C02_FGFR_ACT104_Results.csv"),
  group = c("Ctrl","Ctrl","Ctrl","Ctrl",
            "BR223_EGFR","BR223_EGFR","BR223_EGFR",
            "BR223_IGF1R","BR223_IGF1R","BR223_IGF1R","BR223_IGF1R","BR223_IGF1R",
            "BR223_TrkA","BR223_TrkA","BR223_TrkA",
            "BR223_HER2","BR223_HER2","BR223_HER2",
            "ACT104_EGFR","ACT104_EGFR",
            "ACT104_TrkA","ACT104_TrkA","ACT104_TrkA",
            "ACT104_FGFR"),
  stringsAsFactors = FALSE)

group_order  <- c("Ctrl","BR223_EGFR","BR223_IGF1R","BR223_TrkA","BR223_HER2",
                  "ACT104_EGFR","ACT104_TrkA","ACT104_FGFR")
group_labels <- c("Control","BR223\n+EGFR","BR223\n+IGF1R","BR223\n+TrkA","BR223\n+HER2",
                  "ACT104\n+EGFR","ACT104\n+TrkA","ACT104\n+FGFR")

# --- robust Length reader: Length is the LAST field of each data row.
#     Handles both CSV schemas (with/without a Label column) and the
#     comma/tab mixed-delimiter rows; drops headers and 0-length artifacts.
extract_lengths <- function(path) {
  lines <- readLines(path, warn = FALSE)
  vals <- vapply(lines, function(ln) {
    parts <- strsplit(trimws(ln), "[,\t]")[[1]]
    if (length(parts) < 5) return(NA_real_)
    suppressWarnings(as.numeric(trimws(parts[length(parts)])))
  }, numeric(1), USE.NAMES = FALSE)
  vals <- vals[is.finite(vals)]
  vals[vals > 0]
}

# --- per-file percentage of short (<300) myotubes
files$pct_short <- NA_real_
files$n_tubes   <- NA_integer_
for (i in seq_len(nrow(files))) {
  L <- extract_lengths(file.path(BASE, files$plate[i], files$file[i]))
  files$n_tubes[i]   <- length(L)
  files$pct_short[i] <- 100 * mean(L < THRESH)
}
print(files[, c("group","plate","file","n_tubes","pct_short")], row.names = FALSE)

# --- per-group mean +/- SEM across replicate wells
agg <- do.call(rbind, lapply(group_order, function(g) {
  v <- files$pct_short[files$group == g]; n <- length(v)
  data.frame(group = g, n = n, mean_short = mean(v),
             sem = if (n > 1) sd(v)/sqrt(n) else 0)
}))
agg$mean_long <- 100 - agg$mean_short
write.csv(agg, file.path(BASE, "myotube_length_summary.csv"), row.names = FALSE)
print(agg, row.names = FALSE)

# --- tidy data for a stacked bar (black <300 at bottom, grey >=300 on top)
lab_short <- "< 300 um"; lab_long <- ">= 300 um"
plotdf <- rbind(
  data.frame(group = agg$group, cat = lab_short, pct = agg$mean_short),
  data.frame(group = agg$group, cat = lab_long,  pct = agg$mean_long))
plotdf$group <- factor(plotdf$group, levels = group_order, labels = group_labels)
plotdf$cat   <- factor(plotdf$cat,   levels = c(lab_short, lab_long))

agg$group_f <- factor(agg$group, levels = group_order, labels = group_labels)
agg$nlab    <- paste0("n=", agg$n)

p <- ggplot(plotdf, aes(group, pct, fill = cat)) +
  geom_col(position = position_stack(reverse = TRUE), width = 0.72,
           colour = "white", linewidth = 0.6) +
  geom_errorbar(data = agg,
                aes(x = group_f, ymin = mean_short - sem, ymax = mean_short + sem),
                inherit.aes = FALSE, width = 0.18, colour = "grey30", linewidth = 0.6) +
  geom_text(data = agg, aes(x = group_f, y = 104, label = nlab),
            inherit.aes = FALSE, size = 3, colour = "grey40") +
  scale_fill_manual(values = setNames(c("black", "grey70"), c(lab_short, lab_long)),
                    name = "Myotube length") +
  scale_y_continuous(limits = c(0, 108), breaks = seq(0, 100, 20),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = NULL, y = "Percentage of myotubes (%)",
       title = "Myotube length distribution by novelkin treatment (300 um threshold)") +
  theme_classic(base_size = 12) +
  theme(panel.grid.major.y = element_line(colour = "grey92"),
        legend.position = "right",
        plot.title = element_text(size = 12, hjust = 0.5))

ggsave(file.path(BASE, "myotube_length_by_novelkin_R.png"), p,
       width = 10, height = 6, dpi = 200)
cat("\nSaved chart and summary CSV to", BASE, "\n")
