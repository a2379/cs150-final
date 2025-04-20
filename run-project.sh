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

# Start the backend
cd backend
./project.py &
BACKEND_PID=$!
echo "Backend running with PID $BACKEND_PID"
xdg-open http://localhost:5173
cd ..

# Start the frontend
cd frontend
pnpm run dev &
FRONTEND_PID=$!
echo "Frontend running with PID $FRONTEND_PID"
cd ..

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
