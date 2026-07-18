# Sync project to HiPerGator using PowerShell
# Usage: .\sync_to_hpg.ps1 [-Apply] [-Checksum] [-Help]

param(
    [switch]$Apply,
    [switch]$Checksum,
    [switch]$Help,
    [string]$RemoteUser = "vinhnguyen1",
    [string]$RemoteHost = "rsync.rc.ufl.edu",
    [string]$RemoteDir = "/blue/thai/vinhnguyen1/ia-unforeseen-70b/",
    [string]$LocalDir = "."
)

# Handle both --flag and -flag format
$args | ForEach-Object {
    if ($_ -eq "--apply" -or $_ -eq "-apply") { $Apply = $true }
    if ($_ -eq "--checksum" -or $_ -eq "-checksum") { $Checksum = $true }
    if ($_ -eq "--help" -or $_ -eq "-help" -or $_ -eq "-h") { $Help = $true }
}

# Exclude patterns (similar to bash script)
$ExcludePatterns = @(
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    "outputs",
    "checkpoints",
    "runtime_adapters",
    "models",
    "datasets",
    "logs",
    "wandb",
    "repos",
    "slurm-*.out",
    ".DS_Store",
    "Thumbs.db"
)

function Show-Help {
    Write-Host @"
Sync project to HiPerGator using scp and robocopy

Usage:
  .\sync_to_hpg.ps1 [-Apply] [-Checksum] [-Help]
  .\sync_to_hpg.ps1 --apply --checksum  (also accepts -- format)

Options:
  -Apply or --apply         Actually sync files (default is dry-run)
  -Checksum or --checksum   Use checksum comparison (more accurate, slower)
  -Help or --help           Show this help message
  -RemoteUser               Remote username (default: vinhnguyen1)
  -RemoteHost               Remote host (default: rsync.rc.ufl.edu)
  -RemoteDir                Remote directory (default: /blue/thai/vinhnguyen1/ia-unforeseen-70b/)
  -LocalDir                 Local directory to sync (default: .)

Examples:
  .\sync_to_hpg.ps1
  .\sync_to_hpg.ps1 -Apply
  .\sync_to_hpg.ps1 --apply
  .\sync_to_hpg.ps1 -Apply -Checksum
  .\sync_to_hpg.ps1 --apply --checksum

Notes:
  - Default mode is dry-run (preview only)
  - outputs/ is excluded so remote experiments are never overwritten
  - Requires SSH access to HiPerGator
"@
}

if ($Help) {
    Show-Help
    exit 0
}

# Convert to absolute path
$LocalDir = (Resolve-Path $LocalDir).Path
if (-not (Test-Path $LocalDir)) {
    Write-Error "Local directory not found: $LocalDir"
    exit 1
}

Write-Host "Sync mode: $(if ($Apply) { 'APPLY' } else { 'DRY-RUN' })" -ForegroundColor Cyan
Write-Host "Local dir:  $LocalDir"
Write-Host "Remote:     ${RemoteUser}@${RemoteHost}:${RemoteDir}"
Write-Host ""

# Function to check if path should be excluded
function Test-Exclude {
    param([string]$Path)
    $RelPath = $Path.Substring($LocalDir.Length).TrimStart('\')

    # The project launcher imports this user-maintained dynamic adapter loader
    # directly, so sync it while continuing to exclude the rest of the vendored
    # repository tree.
    if ($RelPath -eq "repos\introspection-adapters\src\finetuning\metalora.py") {
        return $false
    }
    
    foreach ($Pattern in $ExcludePatterns) {
        if ($RelPath -like "*$Pattern*" -or $RelPath -match [regex]::Escape($Pattern)) {
            return $true
        }
    }
    return $false
}

# Get files to sync
$FilesToSync = @()
Get-ChildItem -Path $LocalDir -Recurse -File | ForEach-Object {
    if (-not (Test-Exclude $_.FullName)) {
        $FilesToSync += $_
    }
}

if ($FilesToSync.Count -eq 0) {
    Write-Host "No files to sync." -ForegroundColor Yellow
    exit 0
}

Write-Host "Files to sync: $($FilesToSync.Count)" -ForegroundColor Green
Write-Host ""

# Show file list
foreach ($File in $FilesToSync | Select-Object -First 20) {
    $RelPath = $File.FullName.Substring($LocalDir.Length).TrimStart('\')
    $Size = "{0:N0}" -f $File.Length
    Write-Host "  $RelPath ($Size bytes)"
}

if ($FilesToSync.Count -gt 20) {
    Write-Host "  ... and $($FilesToSync.Count - 20) more files" -ForegroundColor Gray
}

Write-Host ""

if ($Apply) {
    Write-Host "Starting sync..." -ForegroundColor Yellow
    
    # Create temp directory for files to transfer
    $TempSync = Join-Path $env:TEMP "sync_hpg_$(Get-Random)"
    New-Item -ItemType Directory -Path $TempSync -Force | Out-Null
    
    # Track if we're syncing any .slurm files
    $HasSlormFiles = $false
    $SlormFiles = @()
    
    try {
        # Copy files to temp location maintaining structure
        $FilesToSync | ForEach-Object {
            $RelPath = $_.FullName.Substring($LocalDir.Length).TrimStart('\')
            $DestPath = Join-Path $TempSync $RelPath
            $DestDir = Split-Path $DestPath
            
            if (-not (Test-Path $DestDir)) {
                New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
            }
            
            Copy-Item -Path $_.FullName -Destination $DestPath -Force
            Write-Host "  Copied: $RelPath" -ForegroundColor Green
            
            # Track .slurm files
            if ($RelPath -like "*.slurm") {
                $HasSlormFiles = $true
                $SlormFiles += $RelPath
            }
        }
        
        # Transfer to remote using scp
        Write-Host ""
        Write-Host "Uploading to HiPerGator..." -ForegroundColor Cyan
        
        $RemoteTarget = "${RemoteUser}@${RemoteHost}:${RemoteDir}"
        
        # Use scp to transfer all files
        # Note: This requires SSH key setup or will prompt for password
        $scp = "scp -r `"$TempSync\*`" `"$RemoteTarget`""
        
        Write-Host "Running: $scp" -ForegroundColor Gray
        Invoke-Expression $scp
        
        Write-Host ""
        Write-Host "Sync completed successfully!" -ForegroundColor Green
        
        # Warn about .slurm files if they were synced
        if ($HasSlormFiles) {
            Write-Host ""
            Write-Host "⚠️  IMPORTANT: SLURM files detected!" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "The following .slurm files were synced:" -ForegroundColor Yellow
            $SlormFiles | ForEach-Object { Write-Host "  - $_" }
            Write-Host ""
            Write-Host "You MUST convert line endings on HiPerGator:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  ssh ${RemoteUser}@hpg.rc.ufl.edu 'cd ${RemoteDir} && find slurm -name *.slurm -exec dos2unix {} \;'" -ForegroundColor White
            Write-Host ""
            Write-Host "Or manually for each file:" -ForegroundColor Yellow
            Write-Host "  ssh ${RemoteUser}@hpg.rc.ufl.edu 'dos2unix ${RemoteDir}slurm/eval_meta_ia_multi_adapter.slurm'" -ForegroundColor White
            Write-Host ""
        }
    }
    finally {
        # Cleanup temp directory
        Remove-Item -Path $TempSync -Recurse -Force -ErrorAction SilentlyContinue
    }
}
else {
    Write-Host "This is a DRY-RUN. To actually sync, run with --apply flag:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  .\sync_to_hpg.ps1 --apply" -ForegroundColor White
    Write-Host ""
}
