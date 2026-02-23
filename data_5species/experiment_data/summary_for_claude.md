# Data Folder Summary: 5-Species Biofilm Analysis

## Overview
This folder (`data_5species`) contains experimental data and analysis scripts for a 5-species biofilm study. The main focus is on analyzing biofilm growth (volume) and species composition under different conditions.

## Key Analysis Script
**`absolute_volume_analysis.ipynb`**
- **Purpose**: Calculates the **Absolute Volume** of each species over time.
- **Logic**: Combines total biofilm volume with species ratios.
  - Formula: `Absolute Volume = Total Volume (median) * (Species Ratio (median) / 100)`
- **Inputs**:
  - `biofilm_boxplot_data.csv` (Total Volume)
  - `species_distribution_data.csv` (Species Ratios)

## Species Mapping
The datasets use color codes to represent species. The mapping (from `absolute_volume_analysis.ipynb`) is:
- **Blue**: *S. oralis*
- **Green**: *A. naeslundii*
- **Yellow**: *V. dispar*
- **Orange**: *V. parvula*
- **Purple**: *F. nucleatum*
- **Red**: *P. gingivalis*

## Key Data Files & Schema

### 1. `biofilm_boxplot_data.csv` (Total Volume)
Contains boxplot statistics for the total biofilm volume.
- **Columns**: `condition`, `cultivation`, `day`, `median`, `q1`, `q3`, `whisker_low`, `whisker_high`, `fliers`
- **Example**:
  ```csv
  condition,cultivation,day,median,q1,q3,whisker_low,whisker_high,fliers
  Commensal,Static,1,0.42,0.32,0.48,0.18,0.58,0.9
  ```

### 2. `species_distribution_data.csv` (Species Ratios)
Contains statistics for the percentage composition of each species.
- **Columns**: `condition`, `cultivation`, `species`, `day`, `median`, `iqr`, `range`
- **Example**:
  ```csv
  condition,cultivation,species,day,median,iqr,range
  Commensal,Static,Blue,1,18,10,10
  ```

### 3. Other Files
- **`*_all.csv`**: Aggregated data files for specific conditions (e.g., `Commensal_HOBIC_all.csv`).
- **`experimet.ipynb`, `fig.ipynb`**: Additional notebooks for experiments and figure generation.
- **`*.png`**: Generated plots (barplots, boxplots).

## Experimental Conditions
- **Condition**: `Commensal`, `Dysbiotic`
- **Cultivation**: `Static`, `HOBIC`
- **Timepoints (Day)**: 1, 3, 6, 10
