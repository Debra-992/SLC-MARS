# Shared secrets loader for SLC-MARS entry points.
#
# Resolution order, per credential source:
#   1. variables already present in the process environment (PowerShell profile,
#      CI secret store, `-repro` sub-shells) -- these win and the local file is
#      not read at all;
#   2. `secrets.local.ps1` at the repo root (untracked; see secrets.example.ps1).
#
# Neither config2.yaml nor run_route_b.ps1 ever carries a literal credential again.
# Dot-source this file, then call Import-SlcSecrets with the repo root.

function Import-SlcSecrets {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        # quiet: return the source string instead of writing the banner
        [switch]$Quiet
    )

    $names = 'SLC_LLM_BASE_URL', 'SLC_LLM_API_KEY', 'SLC_LLM_MODEL',
             'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD',
             'MONGO_URI', 'MONGO_DB_NAME', 'MONGO_COLLECTION_PAPERS'

    # Never blank out an externally provided value: snapshot which ones are already set.
    $preexisting = @{}
    foreach ($n in $names) { $preexisting[$n] = [Environment]::GetEnvironmentVariable($n) }

    $unset = @($names | Where-Object { [string]::IsNullOrEmpty($preexisting[$_]) })
    $secretsFile = Join-Path $Root 'secrets.local.ps1'

    if ($unset.Count -eq 0) {
        $source = 'environment'
    }
    elseif (Test-Path $secretsFile) {
        . $secretsFile
        $source = 'secrets.local.ps1'
    }
    else {
        throw ("Missing credentials: {0}`n" -f ($unset -join ', ')) +
              "Neither the process environment nor '$secretsFile' provides them.`n" +
              "Fix: Copy-Item '$Root\secrets.example.ps1' '$secretsFile' and fill in real values,`n" +
              "     or export the variables yourself before launching (e.g. from your shell profile)."
    }

    # Any name still empty after the above means secrets.local.ps1 is incomplete.
    $still = @($names | Where-Object { [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($_)) })
    if ($still.Count -gt 0) {
        throw ("secrets source '$source' did not define: {0}" -f ($still -join ', '))
    }

    # Values supplied by the environment must not be silently overwritten by the file.
    foreach ($n in $names) {
        if (-not [string]::IsNullOrEmpty($preexisting[$n])) {
            [Environment]::SetEnvironmentVariable($n, $preexisting[$n])
        }
    }

    if (-not $Quiet) {
        Write-Host ("[secrets] loaded from {0}; api_key={1}, neo4j={2}, mongo={3}" -f $source,
            ([Environment]::GetEnvironmentVariable('SLC_LLM_API_KEY').Length -gt 8),
            ([Environment]::GetEnvironmentVariable('NEO4J_PASSWORD').Length -gt 0),
            ([Environment]::GetEnvironmentVariable('MONGO_URI').Length -gt 10))
    }
    return $source
}
