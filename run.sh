#!/bin/bash
echo "🚀 Starting InsightMed Chatbot System..."

# Run Backend in conda environment
echo "🟢 Starting FastAPI Backend..."
source ./venv/bin/activate
python app.py &
BACKEND_PID=$!

echo "🟢 Starting React Frontend..."
cd frontend
npm run dev -- --port 3000 &
FRONTEND_PID=$!

echo "====================================="
echo "✅ Both Backend (8000) and Frontend (3000) are running."
echo "🟢 UI: http://localhost:3000"
echo "🛑 Press Ctrl+C to stop both processes."
echo "====================================="

# Trap ctrl-c and call cleanup
trap cleanup INT

function cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

wait $BACKEND_PID $FRONTEND_PID
