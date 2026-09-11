# Export the zjsearch theme changes (client sources + server data templates)
# as a single patch to apply on another machine. PowerShell twin of
# make-patch.sh - keep the two in sync.
#
#   powershell -ExecutionPolicy Bypass -File make-patch.ps1 [-Base <ref>] [-Out <file>]
#   # on the target machine:
#   git apply --check zjsearch-theme.patch && git apply zjsearch-theme.patch
#   cd client/zjsearch ; npm install ; npm run build
#
# Default base: the fork's master branch (falls back to origin/master, then
# d834a52), so the patch carries the whole theme delta relative to master.
# Only client/zjsearch and searx/templates/zjsearch are included; both are
# required - the map-page fix lives in the data templates, not the client.

param(
    [string]$Base = "",
    [string]$Out = "zjsearch-theme.patch"
)

$ErrorActionPreference = "Stop"

if (-not $Base) {
    foreach ($ref in @("master", "origin/master", "origin/main", "d834a52")) {
        & git rev-parse -q --verify "$ref" *> $null
        if ($LASTEXITCODE -eq 0) { $Base = $ref; break }
    }
}

Set-Location (Join-Path $PSScriptRoot "..\..")

# byte-exact redirect through cmd: PowerShell's own ">" re-encodes git output
# (UTF-16 / BOM), which git apply would reject
cmd /c "git diff --text $Base HEAD -- client/zjsearch searx/templates/zjsearch > `"$Out`""

$content = Get-Content $Out -Raw
$files = ([regex]::Matches($content, "(?m)^diff --git")).Count
if ($files -eq 0) {
    Write-Error "no changes between $Base and HEAD - nothing to patch"
    exit 1
}
$head = (& git rev-parse --short HEAD).Trim()
Write-Host "wrote $Out : $files files, $Base..$head"
Select-String -Path $Out -Pattern "^diff --git" | ForEach-Object {
    Write-Host ("  " + ($_.Line -replace "^diff --git a/", "" -replace " b/.*$", ""))
}
