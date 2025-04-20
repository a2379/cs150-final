#!/bin/bash

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip3 install -r requirements.txt
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
pnpm install
cd ..

# Function to kill any previous runs using port 5001
kill_port_5001() {
  PORT=5001
  PID=$(lsof -ti tcp:$PORT)
  if [ -n "$PID" ]; then
    echo "Killing process on port $PORT (PID: $PID)..."
    kill -9 $PID
  fi
}
kill_port_5001

# Start the backend
cd backend
./project.py &
BACKEND_PID=$!
echo "Backend running with PID $BACKEND_PID"
cd ..

# Start the frontend
cd frontend
pnpm run dev &
FRONTEND_PID=$!
echo "Frontend running with PID $FRONTEND_PID"
cd ..

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
