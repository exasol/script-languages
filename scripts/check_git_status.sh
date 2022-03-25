#!/bin/bash

#Prints git status and returns > 0 if working tree is dirty! If working tree is clean, it returns 0.

git status --porcelain=v1 -uno
git diff --cached; git diff --cached --summary;
[ -z "$(git status --porcelain=v1 -uno 2>/dev/null)" ]

