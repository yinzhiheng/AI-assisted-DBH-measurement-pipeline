# Read the XLSX package directly using Windows/.NET; Excel is not required.
# Plot labels use Centimeters. Natural labels use Mixed: (0,1) meters, [10,1000) cm.
# Values in the gap are ambiguous and require an explicit unit mode.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$InputXlsx,
    [Parameter(Mandatory=$true)][string]$OutputCsv,
    [ValidateSet('Mixed','Meters','Centimeters')][string]$UnitMode = 'Mixed'
)
$ErrorActionPreference = 'Stop'

function Read-ZipXml($Archive, [string]$Part) {
    $entry = $Archive.GetEntry($Part)
    if ($null -eq $entry) { throw "Missing XLSX part: $Part" }
    $reader = [IO.StreamReader]::new($entry.Open())
    try {
        $doc = [Xml.XmlDocument]::new()
        $doc.XmlResolver = $null
        $doc.LoadXml($reader.ReadToEnd())
        return ,$doc
    } finally { $reader.Dispose() }
}

function Get-CellText($Cell, $SharedStrings) {
    if ($Cell.GetAttribute('t') -eq 'inlineStr') {
        return (($Cell.SelectNodes('./*[local-name()="is"]//*[local-name()="t"]') |
            ForEach-Object { $_.InnerText }) -join '')
    }
    $value = $Cell.SelectSingleNode('./*[local-name()="v"]')
    if ($null -eq $value) { return '' }
    if ($Cell.GetAttribute('t') -eq 's') {
        $index = [int]$value.InnerText
        if ($index -lt 0 -or $index -ge $SharedStrings.Count) { throw 'Invalid shared-string index' }
        return $SharedStrings[$index]
    }
    return $value.InnerText
}

$archive = $null
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($InputXlsx)
    $book = Read-ZipXml $archive 'xl/workbook.xml'
    $sheet = $book.SelectSingleNode('//*[local-name()="sheet" and @name="Labels"]')
    if ($null -eq $sheet) { throw 'Workbook has no Labels worksheet' }
    $relationId = $sheet.GetAttribute('id', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
    $relationships = Read-ZipXml $archive 'xl/_rels/workbook.xml.rels'
    $relation = @($relationships.DocumentElement.ChildNodes | Where-Object {
        $_.GetAttribute('Id') -eq $relationId
    })
    if ($relation.Count -ne 1 -or $relation[0].GetAttribute('TargetMode') -eq 'External') {
        throw 'Cannot resolve Labels worksheet'
    }
    $target = $relation[0].GetAttribute('Target').Replace('\','/')
    $sheetPart = if ($target.StartsWith('/')) { $target.TrimStart('/') } else { 'xl/' + $target }
    $worksheet = Read-ZipXml $archive $sheetPart
    $shared = @()
    if ($null -ne $archive.GetEntry('xl/sharedStrings.xml')) {
        $strings = Read-ZipXml $archive 'xl/sharedStrings.xml'
        $shared = @($strings.DocumentElement.ChildNodes | ForEach-Object {
            (($_.SelectNodes('.//*[local-name()="t"]') | ForEach-Object { $_.InnerText }) -join '')
        })
    }
    $rows = @($worksheet.SelectNodes('//*[local-name()="sheetData"]/*[local-name()="row"]'))
    if ($rows.Count -lt 1) { throw 'Labels worksheet is empty' }
    $columns = @{}
    foreach ($cell in $rows[0].SelectNodes('./*[local-name()="c"]')) {
        $header = (Get-CellText $cell $shared).Trim()
        if ($header) { $columns[$header] = $cell.GetAttribute('r') -replace '\d','' }
    }
    if (-not $columns.ContainsKey('uniqueName') -or -not $columns.ContainsKey('trueDiameter')) {
        throw 'Labels must contain uniqueName and trueDiameter columns'
    }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $records = [Collections.Generic.List[object]]::new()
    $culture = [Globalization.CultureInfo]::InvariantCulture
    foreach ($row in $rows | Select-Object -Skip 1) {
        $id = ''; $raw = ''
        foreach ($cell in $row.SelectNodes('./*[local-name()="c"]')) {
            $col = $cell.GetAttribute('r') -replace '\d',''
            if ($col -eq $columns['uniqueName']) { $id = (Get-CellText $cell $shared).Trim() }
            if ($col -eq $columns['trueDiameter']) { $raw = (Get-CellText $cell $shared).Trim() }
        }
        if (-not $id -and -not $raw) { continue }
        if (-not $id -or -not $seen.Add($id)) { throw "Missing or duplicate uniqueName at row $($row.GetAttribute('r'))" }
        # A missing label remains blank. Never turn it into a zero diameter.
        $unit = ''; $converted = ''
        if ($raw) {
            $value = [decimal]0
            if (-not [decimal]::TryParse($raw, [Globalization.NumberStyles]::Float, $culture, [ref]$value) -or $value -le 0) {
                throw "Invalid trueDiameter for ${id}: $raw"
            }
            if ($UnitMode -eq 'Meters') { $unit = 'm' }
            elseif ($UnitMode -eq 'Centimeters') { $unit = 'cm' }
            elseif ($value -lt 1) { $unit = 'm' }
            elseif ($value -ge 10 -and $value -lt 1000) { $unit = 'cm' }
            else { throw "Ambiguous trueDiameter units for ${id}: $raw. Choose Meters or Centimeters explicitly." }
            $meters = if ($unit -eq 'cm') { $value / 100 } else { $value }
            $converted = $meters.ToString('G29', $culture)
        }
        $records.Add([pscustomobject][ordered]@{
            uniqueName = $id
            ground_truth_source_value = $raw
            ground_truth_source_unit = $unit
            ground_truth_diameter_m = $converted
        })
    }
    if ($records.Count -eq 0) { throw 'No label records in workbook' }
    $records | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
    exit 0
} catch {
    [IO.File]::WriteAllText($OutputCsv + '.error.txt', $_.Exception.Message)
    exit 1
} finally {
    if ($null -ne $archive) { $archive.Dispose() }
}
