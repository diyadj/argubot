# One step on your laptop: start the ArguBot demo on the SCIS cluster, tunnel it, open the browser.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\start_demo.ps1
#   (optional: -Config configs/config.yaml for the 0.8B model, -Port 8000)
#
# Needs the SMU VPN when off campus. Without an SSH key you type the CS425G3 password twice.
# Stop the demo afterwards with:  ssh CS425G3@10.193.104.102 scancel -n argubot-demo
param([string]$Config = "configs/config_4b.yaml", [int]$Port = 8000)
$Login = "CS425G3@10.193.104.102"

Write-Host "Starting the ArguBot demo on the cluster (first time: the model takes a few minutes to load)..."
$output = ssh $Login "cd ~/argubot && bash scripts/start_demo.sh $Config $Port"
$output | ForEach-Object { Write-Host $_ }
$ready = $output | Select-String -Pattern '^READY ' | Select-Object -Last 1
if (-not $ready) {
    Write-Host "The demo did not start. Read the messages above." -ForegroundColor Red
    exit 1
}
$parts = $ready.Line.Split(' ')
$node = $parts[1]
$job = $parts[2]

Write-Host ""
Write-Host "Demo is up on $node (job $job). Opening the tunnel on port $Port."
Write-Host "Your browser opens in a few seconds; if the page does not load, refresh it once the tunnel is connected."
Write-Host "Keep this window open while you present. Ctrl+C closes the tunnel (the demo keeps running)."
Write-Host "To stop the demo and free the GPU: ssh $Login scancel $job"
Start-Job -ScriptBlock { param($p) Start-Sleep -Seconds 8; Start-Process "http://127.0.0.1:$p" } -ArgumentList $Port | Out-Null
ssh -N -L "${Port}:${node}:${Port}" $Login
