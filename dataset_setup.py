"""
dataset_setup.py
================
Acquires an authentic benchmark sample directly from the official CICIoT2023
cybersecurity repository on Hugging Face (baalajimaestro/DDoS-CICIoT2023).

Collects real IoT network flows spanning legitimate traffic and multiple DDoS vectors:
- BenignTraffic (Authentic normal IoT sensor/gateway traffic)
- DDoS-SYN_Flood (Authentic TCP connection table exhaustion)
- DDoS-UDP_Flood (Authentic bandwidth saturation flood)
- DDoS-ICMP_Flood (Authentic ping-of-death volumetric flood)
- DDoS-TCP_Flood (Authentic volumetric TCP flood)

Generates a compact, authentic benchmark dataset: ciciot2023_sample.csv (~15,000 flows).
"""

import os
import sys
import io
import time
import requests
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

CSV_FILENAME = "ciciot2023_sample.csv"

# Authentic CICIoT2023 shards hosted on Hugging Face
AUTHENTIC_SHARDS = {
    "BenignTraffic": {
        "url": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/BenignTraffic.pcap.csv",
        "sample_count": 5000
    },
    "DDoS-SYN_Flood": {
        "url": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/DDoS-SYN_Flood.pcap.csv",
        "sample_count": 2500
    },
    "DDoS-UDP_Flood": {
        "url": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/DDoS-UDP_Flood.pcap.csv",
        "sample_count": 2500
    },
    "DDoS-ICMP_Flood": {
        "url": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/DDoS-ICMP_Flood.pcap.csv",
        "sample_count": 2500
    },
    "DDoS-TCP_Flood": {
        "url": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/DDoS-TCP_Flood.pcap.csv",
        "sample_count": 2500
    }
}


def download_authentic_ciciot2023(output_path: str = CSV_FILENAME) -> pd.DataFrame:
    """
    Streams and samples authentic network flows directly from CICIoT2023 shards.
    """
    print("[*] Acquiring authentic CICIoT2023 dataset from public benchmark repository...")
    dfs = []
    total_requested = sum(cfg["sample_count"] for cfg in AUTHENTIC_SHARDS.values())

    for label_name, cfg in AUTHENTIC_SHARDS.items():
        url = cfg["url"]
        count = cfg["sample_count"]
        print(f"    --> Streaming {count:,} genuine flows for [{label_name}]...")

        try:
            resp = requests.get(url, stream=True, timeout=30)
            if resp.status_code != 200:
                raise ConnectionError(f"HTTP {resp.status_code} while fetching {url}")

            lines = []
            for chunk in resp.iter_lines(decode_unicode=True):
                if chunk:
                    lines.append(chunk)
                    if len(lines) > count:
                        break

            sub_df = pd.read_csv(io.StringIO("\n".join(lines)))
            sub_df["label"] = label_name
            dfs.append(sub_df)
            print(f"        [+] Successfully ingested {len(sub_df):,} flows ({len(sub_df.columns)} features).")

        except Exception as e:
            print(f"        [!] Warning: Error streaming {label_name}: {e}")
            raise

    combined_df = pd.concat(dfs, ignore_index=True)
    # Shuffle flows deterministically
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    combined_df.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[+] Authentic CICIoT2023 sample saved to '{output_path}'")
    print(f"[*] Total Flows: {combined_df.shape[0]:,} | Features: {combined_df.shape[1]} | File Size: {file_size_mb:.2f} MB")
    print("\nAuthentic Class Distribution:")
    print(combined_df["label"].value_counts().to_string())

    return combined_df


def ensure_dataset(output_path: str = CSV_FILENAME, force_refresh: bool = False) -> str:
    """
    Ensures an authentic CICIoT2023 dataset exists locally.
    """
    if not force_refresh and os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
        # Check if existing dataset has authentic columns
        try:
            head_df = pd.read_csv(output_path, nrows=5)
            if "Time_To_Live" in head_df.columns and "Header_Length" in head_df.columns:
                print(f"[✓] Authentic CICIoT2023 dataset already present at '{output_path}' ({os.path.getsize(output_path) / 1e6:.2f} MB)")
                return output_path
        except Exception:
            pass

    download_authentic_ciciot2023(output_path)
    return output_path


if __name__ == "__main__":
    download_authentic_ciciot2023(CSV_FILENAME)
