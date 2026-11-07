#!/bin/bash
# Start the demo job on the cluster (or reuse one that is already running) and wait until the
# model has loaded. The last line it prints is "READY <node> <job id>", which
# scripts/windows/start_demo.ps1 reads to open the tunnel.
#
#   bash scripts/start_demo.sh [config] [port]
CONFIG=${1:-configs/config_4b.yaml}
PORT=${2:-8000}
cd "$(dirname "$0")/.." || exit 1

job_line() { squeue -u "$USER" -n argubot-demo -h -o "%i %T %N" | head -n 1; }

LINE=$(job_line)
if [ -z "$LINE" ]; then
    echo "Submitting the demo job ($CONFIG, port $PORT)..."
    sbatch scripts/violet_demo.sbatch "$CONFIG" "$PORT" || { echo "Could not submit the job (is the queue full?)"; exit 1; }
    sleep 3
else
    echo "Reusing the demo job that is already there: $LINE"
fi

for i in $(seq 1 240); do                     # up to 20 minutes
    LINE=$(job_line)
    [ -n "$LINE" ] || { echo "The demo job ended. Check the newest $USER.demo.*.out log."; exit 1; }
    read -r JOB STATE NODE <<< "$LINE"
    LOG="$USER.demo.$JOB.out"
    if [ "$STATE" = "RUNNING" ] && [ -f "$LOG" ] && grep -q "ArguBot demo" "$LOG"; then
        echo "READY $NODE $JOB"
        exit 0
    fi
    [ $((i % 6)) -eq 1 ] && echo "Waiting: job $JOB is $STATE${NODE:+ on $NODE}, model loading can take a few minutes..."
    sleep 5
done
echo "Gave up after 20 minutes. Check: myqueue and the $USER.demo.*.out log."
exit 1
