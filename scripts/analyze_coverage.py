#!/usr/bin/env python3
"""
Tenzor Coverage Analysis Tool
Analyzes gcov output to identify coverage gaps and generate actionable reports
"""

import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class CoverageAnalyzer:
    def __init__(self, project_root: str, build_dir: str):
        self.project_root = Path(project_root)
        self.build_dir = Path(build_dir)
        self.source_dir = self.project_root / "src"

        self.file_coverage = {}
        self.total_lines = 0
        self.covered_lines = 0
        self.uncovered_files = []

    def run_gcov(self):
        """Run gcov on all .gcda files"""
        print("Running gcov on coverage data...")

        gcda_files = list(self.build_dir.rglob("*.gcda"))
        print(f"Found {len(gcda_files)} .gcda files")

        for gcda_file in gcda_files:
            try:
                # Run gcov
                result = subprocess.run(
                    ["gcov", "-o", str(gcda_file.parent), str(gcda_file.stem)],
                    cwd=str(gcda_file.parent),
                    capture_output=True,
                    text=True
                )
            except Exception as e:
                continue

    def parse_gcov_files(self):
        """Parse .gcov files to extract coverage data"""
        print("Parsing gcov output...")

        gcov_files = list(self.build_dir.rglob("*.gcov"))
        print(f"Found {len(gcov_files)} .gcov files")

        for gcov_file in gcov_files:
            # Skip system headers
            if "/usr/include" in str(gcov_file):
                continue

            # Extract filename from gcov file
            source_file = self.extract_source_name(gcov_file)
            if not source_file:
                continue

            # Parse coverage data
            total, covered, uncovered_lines = self.parse_gcov_file(gcov_file)

            if total > 0:
                coverage_pct = (covered / total) * 100 if total > 0 else 0
                self.file_coverage[source_file] = {
                    'total': total,
                    'covered': covered,
                    'coverage': coverage_pct,
                    'uncovered_lines': uncovered_lines
                }

                self.total_lines += total
                self.covered_lines += covered

    def extract_source_name(self, gcov_file: Path) -> str:
        """Extract the original source file name from .gcov file"""
        try:
            with open(gcov_file, 'r') as f:
                for line in f:
                    if line.startswith("        -:    0:Source:"):
                        source = line.split("Source:")[-1].strip()
                        # Normalize path
                        if source.startswith("/"):
                            return source
                        else:
                            return str((self.project_root / source).resolve())
        except:
            pass
        return ""

    def parse_gcov_file(self, gcov_file: Path) -> Tuple[int, int, List[int]]:
        """Parse a single .gcov file"""
        total_lines = 0
        covered_lines = 0
        uncovered_lines = []

        try:
            with open(gcov_file, 'r') as f:
                for line in f:
                    # Skip header lines
                    if line.strip().startswith("-:"):
                        continue

                    # Match execution count
                    match = re.match(r'\s*(\d+|\#\#\#\#\#):\s*(\d+):', line)
                    if match:
                        exec_count = match.group(1)
                        line_num = int(match.group(2))

                        if exec_count == "#####":
                            # Uncovered line
                            total_lines += 1
                            uncovered_lines.append(line_num)
                        elif exec_count.isdigit() and int(exec_count) > 0:
                            # Covered line
                            total_lines += 1
                            covered_lines += 1
        except:
            pass

        return total_lines, covered_lines, uncovered_lines

    def generate_report(self) -> str:
        """Generate a detailed coverage report"""
        report = []
        report.append("=" * 80)
        report.append("TENZOR COVERAGE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall stats
        overall_pct = (self.covered_lines / self.total_lines * 100) if self.total_lines > 0 else 0
        report.append(f"Overall Coverage: {overall_pct:.2f}%")
        report.append(f"Total Lines: {self.total_lines}")
        report.append(f"Covered Lines: {self.covered_lines}")
        report.append(f"Uncovered Lines: {self.total_lines - self.covered_lines}")
        report.append("")

        # Sort files by coverage
        sorted_files = sorted(
            self.file_coverage.items(),
            key=lambda x: x[1]['coverage']
        )

        # Low coverage files (< 50%)
        report.append("=" * 80)
        report.append("LOW COVERAGE FILES (< 50%)")
        report.append("=" * 80)
        report.append("")

        low_coverage = [f for f in sorted_files if f[1]['coverage'] < 50]
        if low_coverage:
            for file_path, data in low_coverage[:20]:
                rel_path = self.get_relative_path(file_path)
                report.append(f"{data['coverage']:5.1f}% - {rel_path}")
                report.append(f"         ({data['covered']}/{data['total']} lines)")
        else:
            report.append("No files with coverage < 50%")

        report.append("")

        # Medium coverage files (50-80%)
        report.append("=" * 80)
        report.append("MEDIUM COVERAGE FILES (50-80%)")
        report.append("=" * 80)
        report.append("")

        medium_coverage = [f for f in sorted_files if 50 <= f[1]['coverage'] < 80]
        if medium_coverage:
            for file_path, data in medium_coverage[:20]:
                rel_path = self.get_relative_path(file_path)
                report.append(f"{data['coverage']:5.1f}% - {rel_path}")
        else:
            report.append("No files with coverage 50-80%")

        report.append("")

        # Top coverage files
        report.append("=" * 80)
        report.append("TOP 10 BEST COVERAGE FILES")
        report.append("=" * 80)
        report.append("")

        for file_path, data in sorted_files[-10:]:
            rel_path = self.get_relative_path(file_path)
            report.append(f"{data['coverage']:5.1f}% - {rel_path}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def get_relative_path(self, file_path: str) -> str:
        """Get path relative to project root"""
        try:
            return str(Path(file_path).relative_to(self.project_root))
        except:
            return file_path

    def save_report(self, filename: str):
        """Save report to file"""
        report = self.generate_report()
        output_path = self.project_root / "coverage_report" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\nReport saved to: {output_path}")
        return output_path


def main():
    project_root = str(Path(__file__).resolve().parent.parent)
    build_dir = str(Path(project_root) / "build")

    analyzer = CoverageAnalyzer(project_root, build_dir)

    # Run gcov
    analyzer.run_gcov()

    # Parse gcov output
    analyzer.parse_gcov_files()

    # Generate and display report
    print("\n" + analyzer.generate_report())

    # Save report
    analyzer.save_report("coverage_analysis.txt")


if __name__ == "__main__":
    main()
