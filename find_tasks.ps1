$outFile = "C:\Users\joshu\OneDrive\AIBot\Improved\tasks_found.txt"
$tasks = Get-ScheduledTask | Where-Object {$_.Actions.Execute -like '*python*'}
$output = @()
$tasks | ForEach-Object {
    $output += "Task: $($_.TaskName)"
    $output += "Path: $($_.TaskPath)"
    $output += "State: $($_.State)"
    $output += "Action: $($_.Actions.Execute) $($_.Actions.Arguments)"
    $output += "---"
}
if ($tasks.Count -eq 0) {
    $output += "No Python scheduled tasks found"
}
$output | Out-File -FilePath $outFile -Encoding UTF8
Write-Output "Done - check tasks_found.txt"
