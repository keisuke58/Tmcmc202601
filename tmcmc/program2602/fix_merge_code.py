# Fixed merge code for the notebook
# Replace the merge section (lines 474-487) with this:

# Handle potential column name conflicts (both might have 'beta' column)
if "beta" in stage_summary.columns and "beta" in beta_schedule_adjusted.columns:
    # Use beta from beta_schedule, drop stage_summary's beta to avoid conflict
    merged_logL = pd.merge(
        stage_summary.drop(columns=["beta"], errors="ignore"),
        beta_schedule_adjusted[["model", "chain", "stage", "beta"]],
        on=["model", "chain", "stage"],
        how="inner",
    )
elif "beta" in beta_schedule_adjusted.columns:
    # beta only in beta_schedule_adjusted, ensure it's included
    merged_logL = pd.merge(
        stage_summary,
        beta_schedule_adjusted[["model", "chain", "stage", "beta"]],
        on=["model", "chain", "stage"],
        how="inner",
    )
else:
    # beta not in beta_schedule_adjusted, merge all columns
    merged_logL = pd.merge(
        stage_summary, beta_schedule_adjusted, on=["model", "chain", "stage"], how="inner"
    )
    # Check if beta exists after merge
    if "beta" not in merged_logL.columns:
        raise KeyError(
            "'beta' column not found in merged DataFrame. Check that beta_schedule_adjusted contains 'beta' column."
        )

# Sort by beta for better visualization
if "beta" in merged_logL.columns:
    merged_logL = merged_logL.sort_values("beta").reset_index(drop=True)
else:
    print("Warning: 'beta' column not found, skipping sort by beta")
