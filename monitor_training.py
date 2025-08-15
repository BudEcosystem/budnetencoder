#!/usr/bin/env python3
"""
Real-time training monitor for BitNet-QDyT-v2 training.
Shows GPU utilization, memory usage, and training progress.
"""

import time
import subprocess
import os
import re
from pathlib import Path

def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(gpu_util),
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temperature': int(temp)
            }
    except:
        pass
    return None

def get_training_progress():
    """Get training progress from log file."""
    log_file = Path('training_output.log')
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        progress_info = {
            'status': 'Unknown',
            'current_step': 0,
            'current_epoch': 0,
            'loss': None,
            'lr': None,
            'stage': None
        }
        
        # Look for the latest progress information
        for line in reversed(lines):
            # Look for epoch progress
            if 'Epoch ' in line and 'loss' in line.lower():
                epoch_match = re.search(r'Epoch (\d+)', line)
                loss_match = re.search(r'loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
                if epoch_match:
                    progress_info['current_epoch'] = int(epoch_match.group(1))
                if loss_match:
                    progress_info['loss'] = float(loss_match.group(1))
                progress_info['status'] = 'Training'
                break
            
            # Look for step progress
            step_match = re.search(r'Step (\d+)', line)
            if step_match:
                progress_info['current_step'] = int(step_match.group(1))
                progress_info['status'] = 'Training'
        
        # Check if data is loading
        if any('Generating' in line or 'Tokenizing' in line for line in lines[-10:]):
            progress_info['status'] = 'Loading Data'
        
        # Check if training completed
        if any('Training complete' in line for line in lines[-5:]):
            progress_info['status'] = 'Completed'
        
        return progress_info
    except:
        return None

def is_training_running():
    """Check if training process is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'python3 train.py'], capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip()
    except:
        return False

def print_status():
    """Print current training status."""
    os.system('clear')
    print("="*60)
    print("BitNet-QDyT-v2 Training Monitor")
    print("="*60)
    
    # Check if training is running
    training_pid = is_training_running()
    if training_pid:
        print(f"🟢 Training Status: RUNNING (PID: {training_pid.strip()})")
    else:
        print("🔴 Training Status: NOT RUNNING")
    
    print()
    
    # GPU Stats
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print("📊 GPU Statistics (RTX 3080):")
        print(f"   Utilization: {gpu_stats['gpu_util']}%")
        print(f"   Memory: {gpu_stats['mem_used']}/{gpu_stats['mem_total']} MB ({gpu_stats['mem_used']/gpu_stats['mem_total']*100:.1f}%)")
        print(f"   Temperature: {gpu_stats['temperature']}°C")
    else:
        print("❌ GPU Stats: Not available")
    
    print()
    
    # Training Progress
    progress = get_training_progress()
    if progress:
        print("🎯 Training Progress:")
        print(f"   Status: {progress['status']}")
        if progress['current_epoch'] > 0:
            print(f"   Epoch: {progress['current_epoch']}/15")
        if progress['current_step'] > 0:
            print(f"   Step: {progress['current_step']}")
        if progress['loss'] is not None:
            print(f"   Loss: {progress['loss']:.4f}")
        if progress['lr'] is not None:
            print(f"   Learning Rate: {progress['lr']:.2e}")
        if progress['stage'] is not None:
            print(f"   Stage: {progress['stage']}")
    else:
        print("❌ Training Progress: Not available")
    
    print()
    
    # Recent logs
    log_file = Path('training_output.log')
    if log_file.exists():
        print("📝 Recent Log Entries:")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Show last 3 lines
            for line in lines[-3:]:
                line = line.strip()
                if line and not line.startswith('Generating'):
                    print(f"   {line}")
        except:
            print("   Could not read log file")
    
    print()
    print("="*60)
    print("Press Ctrl+C to stop monitoring")
    print("="*60)

def main():
    """Main monitoring loop."""
    try:
        while True:
            print_status()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()