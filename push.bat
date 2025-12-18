@echo off
REM ============================================
REM  init-and-push.bat
REM  Automate first-time setup & push to GitHub
REM ============================================

REM 1. Check if this is already a git repo, if not: git init
if not exist .git (
    echo No .git folder found. Initializing git repo...
    git init
) else (
    echo Git repo already initialized.
)

REM 2. Ask for the GitHub remote URL
echo(
set /p REMOTE_URL=Enter GitHub repo URL (https://github.com/username/repo.git):

if "%REMOTE_URL%"=="" (
    echo No URL entered. Exiting.
    goto :EOF
)

REM 3. Set or update 'origin' remote
git remote get-url origin >nul 2>&1
IF ERRORLEVEL 1 (
    echo Adding remote 'origin'...
    git remote add origin %REMOTE_URL%
) ELSE (
    echo Updating existing remote 'origin'...
    git remote set-url origin %REMOTE_URL%
)

REM 4. Stage all files
echo(
echo Staging all files...
git add .

REM 5. Commit with message (default: "Initial commit")
echo(
set /p COMMIT_MSG=Commit message (leave blank for "Initial commit"):
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial commit

echo Creating commit: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%"

REM 6. Ensure branch is called 'main'
echo(
echo Ensuring branch name is 'main'...
git branch -M main

REM 7. Push to GitHub (main branch)
echo(
echo Pushing to GitHub (origin/main)...
git push -u origin main

echo(
echo ============================================
echo   Done! Repo is linked and pushed to GitHub.
echo ============================================
pause
