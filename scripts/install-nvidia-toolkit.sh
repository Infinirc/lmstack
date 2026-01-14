#!/bin/bash
# Install NVIDIA Container Toolkit
# Supports Ubuntu/Debian and RHEL/CentOS

set -e

echo "=== Installing NVIDIA Container Toolkit ==="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS"
    exit 1
fi

case $OS in
    ubuntu|debian)
        echo "Detected: $OS"

        # Add NVIDIA GPG key
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        # Add repository
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        # Install
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        ;;

    rhel|centos|fedora|rocky|almalinux)
        echo "Detected: $OS"

        # Add repository
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

        # Install
        sudo yum install -y nvidia-container-toolkit
        ;;

    *)
        echo "Unsupported OS: $OS"
        echo "Please install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
        ;;
esac

# Configure Docker
echo ""
echo "=== Configuring Docker runtime ==="
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
echo ""
echo "=== Verifying installation ==="
if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi > /dev/null 2>&1; then
    echo "✓ NVIDIA Container Toolkit installed successfully!"
    docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
else
    echo "✗ Verification failed. Please check your NVIDIA driver installation."
    exit 1
fi
