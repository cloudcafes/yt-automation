#!/bin/bash

# Move to project directory
cd /root/Desktop/yt-automation || exit

# Pull latest changes
git pull

# Run Python script
python3 /root/Desktop/yt-automation/yt-python/main.py

# Prepare date & time strings
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

# Add and commit changes
git add .
git commit -m "python run + $DATE + $TIME"

# Push to repo
git push