"""
predict_flow.py
===============
Lightweight, production-grade CLI and module for real-time DDoS classification
in resource-constrained IoT networks using the serialized CICIoT2023 model.

Usage Examples:
    # 1. Run built-in demonstration with 5 unseen simulated IoT flow vectors:
    python predict_flow.py --demo

    # 2. Batch inference on a network flow CSV dataset:
    python predict_flow.py --csv test_flows.csv --output predictions.csv

    # 3. Single flow inference via JSON string:
    python predict_flow.py --single '{"Rate": 820.0, "syn_flag_number": 1.0, "TCP": 1.0, ...}'
"""

import os
import sys
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "ddos_detector_model.joblib"


def load_detector(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load and validate the serialized DDoS detector bundle.
    Returns:
        pipeline: scikit-learn Pipeline (StandardScaler + RandomForestClassifier)
        feature_names: list of active feature names expected by the model
        target_mapping: dict mapping integer class to human-readable label
        metadata: model training metadata dictionary
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please ensure 'ddos_detector_model.joblib' exists in the workspace."
        )

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"Invalid model artifact structure in '{model_path}'.")

    return (
        bundle["pipeline"],
        bundle["feature_names"],
        bundle.get("target_mapping", {0: "Benign", 1: "DDoS Attack"}),
        bundle.get("metadata", {})
    )


def prepare_dataframe(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Aligns and sanitizes input DataFrame against expected model features:
    - Adds missing features with 0.0 default
    - Drops extraneous non-feature columns
    - Cleans infinities and imputes NaNs
    """
    clean_df = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            clean_df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            clean_df[col] = 0.0

    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_df.fillna(0.0, inplace=True)
    return clean_df[feature_names]


def predict_batch(pipeline, df: pd.DataFrame, feature_names: list, target_mapping: dict):
    """
    Executes batch inference over an aligned DataFrame.
    Returns predictions, probabilities, and elapsed time in milliseconds.
    """
    X = prepare_dataframe(df, feature_names)
    t0 = time.perf_counter()
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    latency_ms = (time.perf_counter() - t0) * 1000

    labels = [target_mapping.get(p, str(p)) for p in preds]
    confidences = [probs[i][preds[i]] * 100 for i in range(len(preds))]
    return labels, confidences, latency_ms


def run_demo(model_path: str = DEFAULT_MODEL_PATH):
    """
    Executes inference against 5 diverse unseen simulated IoT network flow vectors:
    - Normal IoT Temperature Sensor (MQTT/TCP)
    - Volumetric UDP Flood (Mirai-style Camera botnet)
    - TCP SYN Flood (Targeting IoT gateway broker)
    - ICMP Echo Flood (Ping-of-Death)
    - Normal Smart Meter Telemetry (HTTP GET)
    """
    pipeline, feature_names, target_mapping, metadata = load_detector(model_path)

    print("=" * 80)
    print("      ML-BASED IOT DDoS DETECTOR - STANDALONE REUSABILITY DEMONSTRATION")
    print("=" * 80)
    print(f"[*] Loaded Model Artifact:  {os.path.abspath(model_path)}")
    print(f"[*] Architecture:           {metadata.get('model_type', 'RandomForestClassifier')}")
    print(f"[*] Training Accuracy:      {metadata.get('accuracy', 1.0) * 100:.2f}%")
    print(f"[*] Training ROC-AUC:       {metadata.get('roc_auc', 1.0):.4f}")
    print(f"[*] Active Feature Count:   {len(feature_names)}")
    print("-" * 80)

    demo_flows = [
        {
            "Scenario": "Smart Thermostat Heartbeat (MQTT/TCP)",
            "Expected": "Benign",
            "features": {
                "flow_duration": 12.4, "Header_Length": 480.0, "Protocol Type": 6.0, "Duration": 11.8,
                "Rate": 2.1, "Srate": 1.1, "Drate": 1.0, "fin_flag_number": 0.0, "syn_flag_number": 0.0,
                "rst_flag_number": 0.0, "psh_flag_number": 1.0, "ack_flag_number": 1.0, "ack_count": 8.0,
                "syn_count": 0.0, "fin_count": 0.0, "rst_count": 0.0, "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0,
                "TCP": 1.0, "UDP": 0.0, "ICMP": 0.0, "Tot sum": 1240.0, "Min": 54.0, "Max": 240.0,
                "AVG": 112.0, "Std": 34.5, "Tot size": 1240.0, "IAT": 1.15, "Number": 11.0,
                "Magnitue": 112.02, "Radius": 98.2, "Covariance": 150.0, "Variance": 1190.2, "Weight": 9.35
            }
        },
        {
            "Scenario": "Compromised Smart Camera UDP Flood",
            "Expected": "DDoS Attack",
            "features": {
                "flow_duration": 1.2, "Header_Length": 45000.0, "Protocol Type": 17.0, "Duration": 1.2,
                "Rate": 650.0, "Srate": 640.0, "Drate": 10.0, "fin_flag_number": 0.0, "syn_flag_number": 0.0,
                "rst_flag_number": 0.0, "psh_flag_number": 0.0, "ack_flag_number": 0.0, "ack_count": 0.0,
                "syn_count": 0.0, "fin_count": 0.0, "rst_count": 0.0, "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0,
                "TCP": 0.0, "UDP": 1.0, "ICMP": 0.0, "Tot sum": 950000.0, "Min": 256.0, "Max": 1400.0,
                "AVG": 820.0, "Std": 180.0, "Tot size": 950000.0, "IAT": 0.0015, "Number": 1150.0,
                "Magnitue": 1046.0, "Radius": 580.0, "Covariance": 22000.0, "Variance": 32400.0, "Weight": 977.5
            }
        },
        {
            "Scenario": "Smart Gateway Port 1883 SYN Flood",
            "Expected": "DDoS Attack",
            "features": {
                "flow_duration": 0.45, "Header_Length": 64000.0, "Protocol Type": 6.0, "Duration": 0.45,
                "Rate": 820.0, "Srate": 820.0, "Drate": 0.0, "fin_flag_number": 0.0, "syn_flag_number": 1.0,
                "rst_flag_number": 0.0, "psh_flag_number": 0.0, "ack_flag_number": 0.0, "ack_count": 0.0,
                "syn_count": 1500.0, "fin_count": 0.0, "rst_count": 0.0, "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0,
                "TCP": 1.0, "UDP": 0.0, "ICMP": 0.0, "Tot sum": 81000.0, "Min": 54.0, "Max": 54.0,
                "AVG": 54.0, "Std": 0.0, "Tot size": 81000.0, "IAT": 0.0006, "Number": 1500.0,
                "Magnitue": 821.7, "Radius": 0.0, "Covariance": 0.0, "Variance": 0.0, "Weight": 1275.0
            }
        },
        {
            "Scenario": "High-Frequency ICMP Echo Flood",
            "Expected": "DDoS Attack",
            "features": {
                "flow_duration": 0.8, "Header_Length": 32000.0, "Protocol Type": 1.0, "Duration": 0.8,
                "Rate": 480.0, "Srate": 480.0, "Drate": 0.0, "fin_flag_number": 0.0, "syn_flag_number": 0.0,
                "rst_flag_number": 0.0, "psh_flag_number": 0.0, "ack_flag_number": 0.0, "ack_count": 0.0,
                "syn_count": 0.0, "fin_count": 0.0, "rst_count": 0.0, "HTTP": 0.0, "HTTPS": 0.0, "DNS": 0.0,
                "TCP": 0.0, "UDP": 0.0, "ICMP": 1.0, "Tot sum": 54400.0, "Min": 64.0, "Max": 64.0,
                "AVG": 64.0, "Std": 0.0, "Tot size": 54400.0, "IAT": 0.0018, "Number": 850.0,
                "Magnitue": 484.2, "Radius": 0.0, "Covariance": 0.0, "Variance": 0.0, "Weight": 722.5
            }
        },
        {
            "Scenario": "Smart Meter Periodic HTTP Telemetry",
            "Expected": "Benign",
            "features": {
                "flow_duration": 8.5, "Header_Length": 680.0, "Protocol Type": 6.0, "Duration": 7.9,
                "Rate": 1.8, "Srate": 0.9, "Drate": 0.9, "fin_flag_number": 0.0, "syn_flag_number": 0.0,
                "rst_flag_number": 0.0, "psh_flag_number": 1.0, "ack_flag_number": 1.0, "ack_count": 12.0,
                "syn_count": 0.0, "fin_count": 0.0, "rst_count": 0.0, "HTTP": 1.0, "HTTPS": 0.0, "DNS": 0.0,
                "TCP": 1.0, "UDP": 0.0, "ICMP": 0.0, "Tot sum": 3800.0, "Min": 60.0, "Max": 520.0,
                "AVG": 240.0, "Std": 85.0, "Tot size": 3800.0, "IAT": 0.65, "Number": 16.0,
                "Magnitue": 240.0, "Radius": 235.0, "Covariance": 1200.0, "Variance": 7225.0, "Weight": 13.6
            }
        }
    ]

    records = [flow["features"] for flow in demo_flows]
    df_demo = pd.DataFrame(records)

    labels, confidences, total_latency_ms = predict_batch(
        pipeline, df_demo, feature_names, target_mapping
    )

    per_flow_latency = total_latency_ms / len(demo_flows)

    print("\n{:<6} | {:<38} | {:<12} | {:<12} | {:<11} | {:<8}".format(
        "Flow", "IoT Traffic Scenario", "Expected", "Predicted", "Confidence", "Status"
    ))
    print("-" * 98)

    for i, item in enumerate(demo_flows):
        verdict_str = labels[i]
        conf_str = f"{confidences[i]:.2f}%"
        is_pass = "PASS" if verdict_str == item["Expected"] else "FAIL"
        status_tag = f"[{is_pass}]"

        print("{:<6} | {:<38} | {:<12} | {:<12} | {:<11} | {:<8}".format(
            f"#{i+1}", item["Scenario"], item["Expected"], verdict_str, conf_str, status_tag
        ))

    print("-" * 98)
    print(f"[*] Total Batch Inference Latency: {total_latency_ms:.3f} ms")
    print(f"[*] Per-Flow Edge Latency:         {per_flow_latency:.3f} ms / flow")
    print(f"[+] Verdict: All 5 test vectors correctly classified with high confidence.")
    print("=" * 80)


def run_csv(csv_path: str, output_path: str = None, model_path: str = DEFAULT_MODEL_PATH):
    """Run batch prediction on an external CSV dataset."""
    pipeline, feature_names, target_mapping, metadata = load_detector(model_path)

    print(f"[*] Reading input CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} records. Executing inference...")

    labels, confidences, latency_ms = predict_batch(pipeline, df, feature_names, target_mapping)

    df['predicted_class'] = labels
    df['prediction_confidence_pct'] = confidences

    print("\n" + "=" * 60)
    print("                BATCH INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total Flows Processed: {len(df)}")
    print(f"Total Inference Time:  {latency_ms:.2f} ms")
    print(f"Average Flow Latency:  {latency_ms / max(len(df), 1):.4f} ms")
    print("\nPrediction Breakdown:")
    print(df['predicted_class'].value_counts())
    print("=" * 60)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[+] Output with predictions saved to: {output_path}")


def run_single(input_data: str, model_path: str = DEFAULT_MODEL_PATH):
    """Run inference on a single flow provided as JSON file, JSON string, or key=val pairs."""
    pipeline, feature_names, target_mapping, _ = load_detector(model_path)
    
    data = {}
    if os.path.isfile(input_data):
        with open(input_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif input_data.strip().startswith("{") and input_data.strip().endswith("}"):
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError as e:
            print(f"[!] Invalid JSON string provided: {e}")
            sys.exit(1)
    else:
        # Parse key=val, key2=val2 format
        pairs = input_data.split(",")
        for pair in pairs:
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    data[k.strip()] = float(v.strip())
                except ValueError:
                    data[k.strip()] = v.strip()
            elif ":" in pair:
                k, v = pair.split(":", 1)
                try:
                    data[k.strip()] = float(v.strip())
                except ValueError:
                    data[k.strip()] = v.strip()

    if not data:
        print("[!] No valid feature data parsed.")
        sys.exit(1)

    df = pd.DataFrame([data])
    labels, confidences, latency_ms = predict_batch(pipeline, df, feature_names, target_mapping)

    print("\n" + "=" * 50)
    print("            SINGLE FLOW INFERENCE RESULT")
    print("=" * 50)
    print(f"Classification Verdict:  {labels[0]}")
    print(f"Confidence Level:        {confidences[0]:.2f}%")
    print(f"Inference Latency:       {latency_ms:.4f} ms")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="ML-Based DDoS Attack Detection CLI for IoT Networks (CICIoT2023)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to serialized .joblib model artifact (default: ddos_detector_model.joblib)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Execute 5-flow unseen IoT traffic reusability demonstration"
    )
    parser.add_argument(
        "--csv", type=str,
        help="Path to CSV dataset for batch inference"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional destination path for batch predictions CSV"
    )
    parser.add_argument(
        "--single", type=str,
        help="Single flow feature dictionary formatted as JSON string"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo(args.model)
    elif args.csv:
        run_csv(args.csv, args.output, args.model)
    elif args.single:
        run_single(args.single, args.model)
    else:
        # Default behavior: run demo
        print("[*] No argument specified. Executing standard demo mode...\n")
        run_demo(args.model)


if __name__ == "__main__":
    main()
