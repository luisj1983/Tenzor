#!/usr/bin/env python3
"""
Benchmark Report Generator
==========================
Generate visual reports from benchmark results.

Usage:
    python generate_report.py results/benchmark_20250120_120000.json
    python generate_report.py results/*.json --compare
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_utils import BenchmarkResult, load_results


def generate_markdown_report(results: List[BenchmarkResult], sys_info: Dict, output_path: str):
    """Generate a Markdown report."""

    # Group results by category and device
    grouped = {}
    for r in results:
        key = (r.category, r.device)
        if key not in grouped:
            grouped[key] = {"tenzor": [], "pytorch": []}
        grouped[key][r.framework].append(r)

    lines = [
        "# Tenzor Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## System Information",
        "",
        f"- **Platform:** {sys_info.get('platform', 'Unknown')}",
        f"- **CPU:** {sys_info.get('processor', 'Unknown')}",
        f"- **GPU:** {sys_info.get('gpu_name', 'Not available')}",
        f"- **Python:** {sys_info.get('python_version', 'Unknown')}",
        "",
        "---",
        "",
    ]

    # Summary section
    lines.extend([
        "## Executive Summary",
        "",
    ])

    total_speedups = []
    for (category, device), data in grouped.items():
        tz_results = data["tenzor"]
        pt_results = data["pytorch"]

        if tz_results and pt_results:
            for tz_r in tz_results:
                for pt_r in pt_results:
                    if tz_r.name == pt_r.name:
                        total_speedups.append((tz_r.name, tz_r.speedup_vs(pt_r), device))
                        break

    if total_speedups:
        avg_speedup = sum(s[1] for s in total_speedups) / len(total_speedups)
        faster_count = sum(1 for s in total_speedups if s[1] > 1)

        lines.extend([
            f"- **Average speedup vs PyTorch:** {avg_speedup:.2f}x",
            f"- **Benchmarks where Tenzor is faster:** {faster_count}/{len(total_speedups)} ({100*faster_count/len(total_speedups):.0f}%)",
            "",
        ])

    lines.extend([
        "---",
        "",
    ])

    # Detailed results by category
    for (category, device), data in sorted(grouped.items()):
        tz_results = data["tenzor"]
        pt_results = data["pytorch"]

        lines.extend([
            f"## {category.replace('_', ' ').title()} ({device.upper()})",
            "",
        ])

        if not tz_results:
            lines.extend(["*No results*", ""])
            continue

        # Create comparison table
        lines.extend([
            "| Benchmark | Tenzor (ms) | PyTorch (ms) | Speedup | GFLOPS |",
            "|-----------|-------------|--------------|---------|--------|",
        ])

        for tz_r in tz_results:
            pt_r = None
            for p in pt_results:
                if p.name == tz_r.name:
                    pt_r = p
                    break

            tz_time = f"{tz_r.mean_ms:.3f}"
            pt_time = f"{pt_r.mean_ms:.3f}" if pt_r else "-"
            speedup = f"{tz_r.speedup_vs(pt_r):.2f}x" if pt_r else "-"
            gflops = f"{tz_r.gflops:.1f}" if tz_r.gflops else "-"

            # Add indicator
            if pt_r:
                if tz_r.speedup_vs(pt_r) >= 1.1:
                    indicator = " ✅"
                elif tz_r.speedup_vs(pt_r) <= 0.9:
                    indicator = " ⚠️"
                else:
                    indicator = " ≈"
            else:
                indicator = ""

            lines.append(f"| {tz_r.name} | {tz_time} | {pt_time} | {speedup}{indicator} | {gflops} |")

        lines.append("")

    # Performance analysis section
    lines.extend([
        "---",
        "",
        "## Performance Analysis",
        "",
    ])

    # Find best and worst performers
    if total_speedups:
        sorted_by_speedup = sorted(total_speedups, key=lambda x: x[1], reverse=True)

        lines.extend([
            "### Top 5 Fastest (vs PyTorch)",
            "",
        ])
        for name, speedup, device in sorted_by_speedup[:5]:
            lines.append(f"1. **{name}** ({device}): {speedup:.2f}x faster")

        lines.extend([
            "",
            "### Top 5 Slowest (vs PyTorch)",
            "",
        ])
        for name, speedup, device in sorted_by_speedup[-5:]:
            status = "slower" if speedup < 1 else "faster"
            lines.append(f"1. **{name}** ({device}): {speedup:.2f}x {status}")

    lines.extend([
        "",
        "---",
        "",
        "## Methodology",
        "",
        "- All benchmarks use synchronized timing for accurate GPU measurements",
        "- Warmup iterations: 5-10 (not included in measurements)",
        "- Benchmark iterations: 100-200 (for statistical significance)",
        "- Metrics: Mean, std dev, min, max, P95, P99",
        "- GFLOPS calculated where applicable (for compute-bound operations)",
        "",
        "## Legend",
        "",
        "- ✅ Tenzor is >10% faster than PyTorch",
        "- ≈ Performance is within 10%",
        "- ⚠️ Tenzor is >10% slower than PyTorch",
        "",
        "---",
        "",
        f"*Report generated by Tenzor Benchmark Suite v1.0.0*",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Markdown report saved to: {output_path}")


def generate_csv_report(results: List[BenchmarkResult], output_path: str):
    """Generate a CSV report for spreadsheet analysis."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Name', 'Category', 'Device', 'Framework',
            'Mean (ms)', 'Std Dev (ms)', 'Min (ms)', 'Max (ms)',
            'Median (ms)', 'P95 (ms)', 'P99 (ms)',
            'GFLOPS', 'Bandwidth (GB/s)', 'Iterations'
        ])

        # Data
        for r in results:
            writer.writerow([
                r.name, r.category, r.device, r.framework,
                f"{r.mean_ms:.4f}", f"{r.std_ms:.4f}", f"{r.min_ms:.4f}", f"{r.max_ms:.4f}",
                f"{r.median_ms:.4f}", f"{r.p95_ms:.4f}", f"{r.p99_ms:.4f}",
                f"{r.gflops:.2f}" if r.gflops else "",
                f"{r.bandwidth_gbps:.2f}" if r.bandwidth_gbps else "",
                r.iterations
            ])

    print(f"CSV report saved to: {output_path}")


def generate_html_report(results: List[BenchmarkResult], sys_info: Dict, output_path: str):
    """Generate an HTML report with charts."""

    # Group by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tenzor Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat {{ text-align: center; padding: 20px; background: #f9f9f9; border-radius: 8px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .faster {{ color: #4CAF50; font-weight: bold; }}
        .slower {{ color: #f44336; font-weight: bold; }}
        .chart-container {{ height: 400px; margin: 20px 0; }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Tenzor Benchmark Report</h1>

        <div class="card">
            <h2>System Information</h2>
            <p><strong>Platform:</strong> {sys_info.get('platform', 'Unknown')}</p>
            <p><strong>CPU:</strong> {sys_info.get('processor', 'Unknown')}</p>
            <p><strong>GPU:</strong> {sys_info.get('gpu_name', 'Not available')}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""

    # Calculate summary stats
    speedups = []
    for cat_results in categories.values():
        tz_results = [r for r in cat_results if r.framework == "tenzor"]
        pt_results = [r for r in cat_results if r.framework == "pytorch"]
        for tz_r in tz_results:
            for pt_r in pt_results:
                if tz_r.name == pt_r.name and tz_r.device == pt_r.device:
                    speedups.append(tz_r.speedup_vs(pt_r))

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        faster_pct = 100 * sum(1 for s in speedups if s > 1) / len(speedups)
        max_speedup = max(speedups)
    else:
        avg_speedup = 1.0
        faster_pct = 0
        max_speedup = 1.0

    html += f"""
        <div class="card">
            <h2>Summary</h2>
            <div class="summary">
                <div class="stat">
                    <div class="stat-value">{avg_speedup:.2f}x</div>
                    <div class="stat-label">Avg Speedup vs PyTorch</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{faster_pct:.0f}%</div>
                    <div class="stat-label">Tests Faster Than PyTorch</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{max_speedup:.2f}x</div>
                    <div class="stat-label">Max Speedup</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Total Benchmarks</div>
                </div>
            </div>
        </div>
"""

    # Results tables by category
    for category, cat_results in categories.items():
        tz_results = [r for r in cat_results if r.framework == "tenzor"]
        pt_results = [r for r in cat_results if r.framework == "pytorch"]

        html += f"""
        <div class="card">
            <h2>{category.replace('_', ' ').title()}</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Device</th>
                        <th>Tenzor (ms)</th>
                        <th>PyTorch (ms)</th>
                        <th>Speedup</th>
                        <th>GFLOPS</th>
                    </tr>
                </thead>
                <tbody>
"""

        for tz_r in tz_results:
            pt_r = None
            for p in pt_results:
                if p.name == tz_r.name and p.device == tz_r.device:
                    pt_r = p
                    break

            pt_time = f"{pt_r.mean_ms:.3f}" if pt_r else "-"
            if pt_r:
                speedup = tz_r.speedup_vs(pt_r)
                speedup_class = "faster" if speedup > 1 else "slower"
                speedup_str = f'<span class="{speedup_class}">{speedup:.2f}x</span>'
            else:
                speedup_str = "-"

            gflops = f"{tz_r.gflops:.1f}" if tz_r.gflops else "-"

            html += f"""
                    <tr>
                        <td>{tz_r.name}</td>
                        <td>{tz_r.device}</td>
                        <td>{tz_r.mean_ms:.3f}</td>
                        <td>{pt_time}</td>
                        <td>{speedup_str}</td>
                        <td>{gflops}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
"""

    html += """
        <div class="footer">
            <p>Generated by Tenzor Benchmark Suite v1.0.0</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("input", help="Input JSON file with benchmark results")
    parser.add_argument("--format", "-f", choices=["all", "md", "csv", "html"], default="all",
                        help="Output format")
    parser.add_argument("--output", "-o", help="Output directory", default="reports")

    args = parser.parse_args()

    # Load results
    with open(args.input, 'r') as f:
        data = json.load(f)

    results = [BenchmarkResult(**r) for r in data["results"]]
    sys_info = data.get("system_info", {})

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate base filename
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # Generate reports
    if args.format in ["all", "md"]:
        generate_markdown_report(results, sys_info, f"{args.output}/{base_name}.md")

    if args.format in ["all", "csv"]:
        generate_csv_report(results, f"{args.output}/{base_name}.csv")

    if args.format in ["all", "html"]:
        generate_html_report(results, sys_info, f"{args.output}/{base_name}.html")

    print(f"\nReports generated in: {args.output}/")


if __name__ == "__main__":
    main()
