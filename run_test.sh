#!/bin/bash
# GraphRAG Test Runner Script
# Makes it easy to run GraphRAG tests from the host machine

set -e

echo "================================"
echo "  GraphRAG Test Runner"
echo "================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Check if containers are running
if ! docker ps | grep -q erica-backend; then
    echo "⚠️  Backend container not running"
    echo "Starting services..."
    docker-compose up -d
    echo "Waiting for services to be ready..."
    sleep 5
fi

# Parse arguments
MODE="${1:---default}"

case "$MODE" in
    -i|--interactive)
        echo "🎓 Starting interactive GraphRAG testing..."
        docker exec -it erica-backend python -m test_graphrag --interactive
        ;;
    -m|--multi)
        echo "🧪 Running multi-query test suite..."
        docker exec -it erica-backend python -m test_graphrag --multi
        ;;
    -d|--demo)
        echo "📝 Running demonstration questions..."
        docker exec -it erica-backend python -m test_graphrag --demo
        ;;
    -s|--stats)
        echo "📊 Showing graph statistics..."
        docker exec -it erica-backend python -c "from test_graphrag import display_graph_stats; display_graph_stats()"
        ;;
    --default)
        echo "🔍 Running default test query..."
        docker exec -it erica-backend python -m test_graphrag
        ;;
    -h|--help)
        echo "Usage: ./run_test.sh [OPTIONS] or [QUERY]"
        echo ""
        echo "Options:"
        echo "  -i, --interactive    Interactive mode (ask multiple questions)"
        echo "  -m, --multi          Run multi-query test suite"
        echo "  -d, --demo           Run demonstration questions"
        echo "  -s, --stats          Show graph statistics only"
        echo "  -h, --help           Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_test.sh                              # Default test"
        echo "  ./run_test.sh --interactive                # Interactive mode"
        echo "  ./run_test.sh --demo                       # Demo questions"
        echo "  ./run_test.sh \"What is attention?\"         # Custom query"
        ;;
    *)
        echo "🔍 Running custom query: $*"
        docker exec -it erica-backend python -m test_graphrag "$*"
        ;;
esac

echo ""
echo "✅ Test completed!"
