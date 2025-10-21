#!/usr/bin/env bash
set -euo pipefail

RETRY_SLEEP=1
BATCH_LIMIT=100  # number of runs to fetch per loop (adjust as needed)
CONCURRENCY=5    # number of concurrent deletions (adjust based on rate limits)
DRY_RUN=false    # set to false to actually delete

echo "Dry run mode: $DRY_RUN"

# Function to delete a single run
delete_run() {
  local id=$1
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY] Would delete run id: $id"
  else
    echo "Deleting run id: $id"
    if ! gh run delete "$id"; then
      echo "Delete failed for $id, retrying after $RETRY_SLEEP s..."
      sleep "$RETRY_SLEEP"
      gh run delete "$id" || echo "Second attempt failed for $id"
    fi
  fi
}

while true; do
  # Fetch up to $BATCH_LIMIT run ids
  ids_raw=$(gh run list --limit "$BATCH_LIMIT" --json databaseId --jq '.[].databaseId')

  # Stop if no runs found
  if [[ -z "$ids_raw" ]]; then
    echo "No more runs found. Done."
    break
  fi

  # Convert to array
  read -r -d '' -a ids <<< "$ids_raw" || true

  echo "Found ${#ids[@]} runs"

  # Process deletions concurrently
  for id in "${ids[@]}"; do
    # Run deletion in background
    delete_run "$id" &
    # Control concurrency: wait if too many jobs are running
    while (( $(jobs -r | wc -l) >= CONCURRENCY )); do
      sleep 0.1  # Brief wait to check job status
    done
  done

  # Wait for all background jobs in this batch to complete
  wait

  # If dry-run, break after first batch to inspect output
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. Set DRY_RUN=false to perform deletions."
    break
  fi
done

echo "Script complete."