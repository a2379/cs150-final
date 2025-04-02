#! /bin/sh

cd frontend;
echo "Opening http://localhost:5173 (you may have to refresh the page)\n"
xdg-open http://localhost:5173
pnpm run dev & python3 ../backend/project.py && fg
