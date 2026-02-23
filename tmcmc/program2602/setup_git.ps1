# Git setup script for tmcmc_docs
# Run this script after Git is installed

# Navigate to the directory
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs"

# Initialize git repository (if not already initialized)
if (-not (Test-Path .git)) {
    git init
    Write-Host "Git repository initialized"
} else {
    Write-Host "Git repository already exists"
}

# Add remote repository
git remote remove origin 2>$null
git remote add origin https://github.com/keisuke58/Tmcmc202601.git
Write-Host "Remote 'origin' set to https://github.com/keisuke58/Tmcmc202601.git"

# Check current status
Write-Host "`nCurrent git status:"
git status

Write-Host "`nNext steps:"
Write-Host "1. Review and create .gitignore if needed"
Write-Host "2. Add files: git add ."
Write-Host "3. Commit: git commit -m 'Initial commit'"
Write-Host "4. Push: git push -u origin main (or master, depending on your default branch)"
