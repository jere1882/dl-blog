#!/bin/bash

echo "🧪 Testing Docker build locally..."

# Build the Docker image
echo "Building Docker image..."
docker build -t egyptian-art-analyzer:test .

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful!"
    
    # Test running the container locally
    echo "Testing container locally..."
    docker run --rm -e GOOGLE_API_KEY=test-key egyptian-art-analyzer:test
    
    echo "✅ Docker test completed!"
else
    echo "❌ Docker build failed!"
    exit 1
fi
