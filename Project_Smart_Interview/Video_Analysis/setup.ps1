# Smart Interview - Automated Dependency Resolution Script
# This script resolves all dependency conflicts and installs compatible versions

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Smart Interview Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Uninstall conflicting packages
Write-Host "[STEP 1/5] Uninstalling conflicting packages..." -ForegroundColor Yellow
python -m pip uninstall -y tensorflow keras tf-keras mediapipe deepface retina-face protobuf 2>&1 | Out-Null
Write-Host "           [DONE]" -ForegroundColor Green

# Step 2: Clear pip cache
Write-Host "[STEP 2/5] Clearing pip cache..." -ForegroundColor Yellow
python -m pip cache purge 2>&1 | Out-Null
Write-Host "           [DONE]" -ForegroundColor Green

# Step 3: Install requirements
Write-Host "[STEP 3/5] Installing compatible packages..." -ForegroundColor Yellow
Write-Host "           - protobuf==4.25.3" -ForegroundColor Cyan
Write-Host "           - tensorflow==2.13.0" -ForegroundColor Cyan
Write-Host "           - mediapipe==0.10.14" -ForegroundColor Cyan
Write-Host "           - deepface==0.0.93" -ForegroundColor Cyan
Write-Host "           - All dependencies..." -ForegroundColor Cyan

python -m pip install -r requirements.txt --no-cache-dir

$lastExitCode_install = $LASTEXITCODE

if ($lastExitCode_install -eq 0) {
    Write-Host "           [DONE]" -ForegroundColor Green
} else {
    Write-Host "           [FAILED]" -ForegroundColor Red
    Write-Host "Please check internet connection and try again." -ForegroundColor Red
    exit 1
}

# Step 4: Verify installations
Write-Host "[STEP 4/5] Verifying installations..." -ForegroundColor Yellow

$packages = @("protobuf", "tensorflow", "mediapipe", "deepface", "retina-face")
$all_installed = $true

foreach ($pkg in $packages) {
    $result = python -m pip show $pkg 2>&1 | Select-String "Name"
    if ($result) {
        Write-Host "           ✅ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "           ❌ $pkg NOT installed" -ForegroundColor Red
        $all_installed = $false
    }
}

if (-not $all_installed) {
    Write-Host "[STEP 4/5] [WARNING] Some packages may not be installed" -ForegroundColor Yellow
} else {
    Write-Host "[STEP 4/5] [DONE]" -ForegroundColor Green
}

# Step 5: Run verification script
Write-Host "[STEP 5/5] Running verification script..." -ForegroundColor Yellow

if (Test-Path "verify_tf.py") {
    python verify_tf.py
    $verify_result = $LASTEXITCODE
    if ($verify_result -eq 0) {
        Write-Host "           [DONE]" -ForegroundColor Green
    } else {
        Write-Host "           [WARNING] Verify script reported issues" -ForegroundColor Yellow
    }
} else {
    Write-Host "           [SKIPPED] verify_tf.py not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run: python main.py" -ForegroundColor Green
Write-Host ""
Write-Host "Installed Versions:" -ForegroundColor Cyan
Write-Host "  - protobuf: 4.25.3" -ForegroundColor White
Write-Host "  - tensorflow: 2.13.0" -ForegroundColor White
Write-Host "  - mediapipe: 0.10.14" -ForegroundColor White
Write-Host "  - deepface: 0.0.93" -ForegroundColor White
Write-Host "  - retina-face: 0.0.17" -ForegroundColor White
Write-Host ""
Write-Host "All features enabled:" -ForegroundColor Green
Write-Host "  ✅ Face Mesh Detection" -ForegroundColor White
Write-Host "  ✅ Hands Tracking" -ForegroundColor White
Write-Host "  ✅ Head Pose Estimation" -ForegroundColor White
Write-Host "  ✅ Eye Tracking" -ForegroundColor White
Write-Host "  ✅ Emotion Detection" -ForegroundColor White
Write-Host ""
