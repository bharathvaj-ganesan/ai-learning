#!/usr/bin/env bash
# Create commits spread across a date range, then push the current branch.
#
# Usage:
#   ./random_commits.sh START END [--max N] [--chance PERCENT] [--remote NAME]
#
# Example:
#   ./random_commits.sh 2026-01-01 2026-11-08
#   ./random_commits.sh 2026-01-01 2026-11-08 --max 4 --chance 60 --remote origin
#
# Each day in the range is included with probability PERCENT (default 50).
# An included day gets 1..MAX commits (default 3), at random times.
# Only activity.log is staged, so other working-tree changes stay unstaged.
# Author and committer dates are set per commit. Git config is not changed.
# Push uses the current branch and refuses a missing remote. No force push.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 START END [--max N] [--chance PERCENT] [--remote NAME]" >&2
  exit 1
fi

start=$1
end=$2
shift 2

max_per_day=3
chance=50
remote=origin

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max)
      max_per_day=${2:?--max needs a number}
      shift 2
      ;;
    --chance)
      chance=${2:?--chance needs a percent}
      shift 2
      ;;
    --remote)
      remote=${2:?--remote needs a name}
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! "$start" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || [[ ! "$end" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Dates must be YYYY-MM-DD. Got start=$start end=$end" >&2
  exit 1
fi

if ! date -j -f "%Y-%m-%d" "$start" "+%Y-%m-%d" >/dev/null 2>&1; then
  echo "Invalid start date: $start" >&2
  exit 1
fi
if ! date -j -f "%Y-%m-%d" "$end" "+%Y-%m-%d" >/dev/null 2>&1; then
  echo "Invalid end date: $end" >&2
  exit 1
fi
if [[ "$start" > "$end" ]]; then
  echo "Start date is after end date: $start > $end" >&2
  exit 1
fi
if [[ ! "$max_per_day" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max must be a positive integer" >&2
  exit 1
fi
if [[ ! "$chance" =~ ^[0-9]+$ ]] || (( chance > 100 )); then
  echo "--chance must be an integer from 0 to 100" >&2
  exit 1
fi

git rev-parse --is-inside-work-tree >/dev/null

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" == "HEAD" ]]; then
  echo "Detached HEAD. Check out a branch before running this." >&2
  exit 1
fi

log_file="activity.log"
messages=(
  "wip"
  "notes"
  "update"
  "tweak"
  "checkpoint"
  "progress"
)

day=$start
made=0

while [[ "$day" < "$end" || "$day" == "$end" ]]; do
  if (( RANDOM % 100 < chance )); then
    count=$(( 1 + RANDOM % max_per_day ))
    for (( i = 1; i <= count; i++ )); do
      hour=$(printf "%02d" $(( RANDOM % 16 + 8 )))
      minute=$(printf "%02d" $(( RANDOM % 60 )))
      second=$(printf "%02d" $(( RANDOM % 60 )))
      stamp="$day $hour:$minute:$second"
      msg=${messages[$(( RANDOM % ${#messages[@]} ))]}

      printf '%s %s\n' "$stamp" "$msg" >> "$log_file"
      git add -- "$log_file"
      GIT_AUTHOR_DATE="$stamp" GIT_COMMITTER_DATE="$stamp" \
        git commit -m "$msg" -- "$log_file" >/dev/null
      made=$(( made + 1 ))
    done
  fi
  day=$(date -j -f "%Y-%m-%d" -v+1d "$day" "+%Y-%m-%d")
done

echo "Created $made commits on $branch from $start through $end."

if ! git remote get-url "$remote" >/dev/null 2>&1; then
  echo "No remote named '$remote'. Commits are local only." >&2
  echo "Add one, then run: git push -u $remote $branch" >&2
  exit 1
fi

git push -u "$remote" "$branch"
echo "Pushed $branch to $remote."
