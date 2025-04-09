#! /bin/sh

cd frontend;
pnpm run dev & python3 ../backend/project.py 127.0.0.1 3000 && fg
