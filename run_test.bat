@echo off
REM GraphRAG Test Runner Script for Windows
REM Makes it easy to run GraphRAG tests from the host machine

echo ================================
echo   GraphRAG Test Runner
echo ================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running
    echo Please start Docker Desktop and try again
    exit /b 1
)

REM Check if containers are running
docker ps | findstr erica-backend >nul
if errorlevel 1 (
    echo Backend container not running
    echo Starting services...
    docker-compose up -d
    echo Waiting for services to be ready...
    timeout /t 5 /nobreak >nul
)

REM Parse arguments
set MODE=%1
if "%MODE%"=="" set MODE=--default

if "%MODE%"=="-i" goto interactive
if "%MODE%"=="--interactive" goto interactive
if "%MODE%"=="-m" goto multi
if "%MODE%"=="--multi" goto multi
if "%MODE%"=="-d" goto demo
if "%MODE%"=="--demo" goto demo
if "%MODE%"=="-s" goto stats
if "%MODE%"=="--stats" goto stats
if "%MODE%"=="-h" goto help
if "%MODE%"=="--help" goto help
if "%MODE%"=="--default" goto default
goto custom

:interactive
echo Starting interactive GraphRAG testing...
docker exec -it erica-backend python -m test_graphrag --interactive
goto end

:multi
echo Running multi-query test suite...
docker exec -it erica-backend python -m test_graphrag --multi
goto end

:demo
echo Running demonstration questions...
docker exec -it erica-backend python -m test_graphrag --demo
goto end

:stats
echo Showing graph statistics...
docker exec -it erica-backend python -c "from test_graphrag import display_graph_stats; display_graph_stats()"
goto end

:default
echo Running default test query...
docker exec -it erica-backend python -m test_graphrag
goto end

:help
echo Usage: run_test.bat [OPTIONS] or [QUERY]
echo.
echo Options:
echo   -i, --interactive    Interactive mode (ask multiple questions)
echo   -m, --multi          Run multi-query test suite
echo   -d, --demo           Run demonstration questions
echo   -s, --stats          Show graph statistics only
echo   -h, --help           Show this help message
echo.
echo Examples:
echo   run_test.bat                              # Default test
echo   run_test.bat --interactive                # Interactive mode
echo   run_test.bat --demo                       # Demo questions
echo   run_test.bat "What is attention?"         # Custom query
goto end

:custom
echo Running custom query: %*
docker exec -it erica-backend python -m test_graphrag %*
goto end

:end
echo.
echo Test completed!
