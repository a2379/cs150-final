#! /bin/sh

cd frontend;
pnpm run --dev & python3 ../backend/project.py && fg
