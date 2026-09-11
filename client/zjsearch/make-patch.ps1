# Export the zjsearch theme changes (client sources + server data templates)
# as a single patch to apply on another machine. PowerShell twin of
# make-patch.sh - keep the two in sync.
#
#   powershell -ExecutionPolicy Bypass -File make-patch.ps1 [-Base <ref>] [-Out <file>]
#   # on the target machine:
#   git apply --check zjsearch-theme.patch && git apply zjsearch-theme.patch
#   cd client/zjsearch ; npm install ; npm run build
#
# Default base: master (falls back to origin/master; if neither exists,
# origin/master is fetched from origin first), so the patch carries the whole
# theme delta relative to master. The diff runs against the working tree, so
# uncommitted edits are included (untracked files are not). Only client/zjsearch
# and searx/templates/zjsearch are included; both are required - the map-page
# fix lives in the data templates, not the client.

param(
    [string]$Base = "",
    [string]$Out = "zjsearch-theme.patch"
)

$ErrorActionPreference = "Stop"

if (-not $Base) {
    foreach ($ref in @("master", "origin/master")) {
        & git rev-parse -q --verify "$ref" *> $null
        if ($LASTEXITCODE -eq 0) { $Base = $ref; break }
    }
    if (-not $Base) {
        Write-Host "no master/origin/master locally - fetching origin/master ..."
        # explicit refspec so the remote-tracking ref exists even in
        # single-branch clones (fetching a bare branch name only fills FETCH_HEAD)
        & git fetch --quiet origin "+refs/heads/master:refs/remotes/origin/master"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "error: could not fetch origin/master - check the network, or pass -Base <ref>"
            exit 1
        }
        $Base = "origin/master"
    }
}

Set-Location (Join-Path $PSScriptRoot "..\..")

# byte-exact redirect through cmd: PowerShell's own ">" re-encodes git output
# (UTF-16 / BOM), which git apply would reject
cmd /c "git diff --text $Base -- client/zjsearch searx/templates/zjsearch > `"$Out`""

$content = Get-Content $Out -Raw
$files = ([regex]::Matches($content, "(?m)^diff --git")).Count
if ($files -eq 0) {
    Write-Error "no changes between $Base and the working tree - nothing to patch"
    exit 1
}
Write-Host "wrote $Out : $files files, $Base..worktree"
Select-String -Path $Out -Pattern "^diff --git" | ForEach-Object {
    Write-Host ("  " + ($_.Line -replace "^diff --git a/", "" -replace " b/.*$", ""))
}
