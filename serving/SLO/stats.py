import os
import numpy as np
import argparse

def process_log_file(log_file):
    # Check if the file exists
    if not os.path.isfile(log_file):
        print(f"Error: The file '{log_file}' does not exist or is not a valid file.")
        return

    print(f"Processing {log_file}...")

    latencies = []
    with open(log_file, "r") as file:
        lines = file.readlines()

    # Find the index of "Final Latency Report"
    start_index = None
    for i, line in enumerate(lines):
        if "Final Latency Report" in line:
            start_index = i + 1
            break

    # Extract latencies directly from the next 1000 lines
    if start_index:
        latencies = [float(line.strip()) for line in lines[start_index : start_index + 1000] if line.strip()]

    # Compute statistics
    if latencies:
        total_count = len(latencies)
        p99_latency = np.percentile(latencies, 99)
        count_100 = sum(lat > 120 for lat in latencies)
        count_200 = sum(lat > 240 for lat in latencies)

        ratio_100 = count_100 / total_count if total_count > 0 else 0
        ratio_200 = count_200 / total_count if total_count > 0 else 0

        print(f"  99th Percentile Latency: {p99_latency:.2f} s")
        print(f"  Number of latencies > 120 s: {count_100} ({ratio_100:.2%})")
        print(f"  Number of latencies > 240 s: {count_200} ({ratio_200:.2%})\n")
    else:
        print("  No latency data found after 'Final Latency Report'.\n")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a single log file to extract latency statistics.")
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to the log file."
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the log file
    process_log_file(args.log_file)
