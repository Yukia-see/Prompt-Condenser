#!/usr/bin/env python3
"""Automated Bridge between Gemini and Claude Code.

Handles:
1. Sending messages/reports to Claude by writing `docs/reports/gemini_<name>.md`.
2. Monitoring and receiving reports from Claude (`docs/reports/claude_*.md`) via inotify.
3. Reading latest status and reports.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

REPORTS_DIR = Path("/home/yukia/Documents/Prompt Condensor/docs/reports")


def get_latest_claude_report() -> Path | None:
    """Find the most recently modified report written by Claude."""
    claude_files = list(REPORTS_DIR.glob("claude_*.md"))
    if not claude_files:
        return None
    claude_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return claude_files[0]


def get_latest_gemini_report() -> Path | None:
    """Find the most recently modified report written by Gemini."""
    gemini_files = list(REPORTS_DIR.glob("gemini_*.md"))
    if not gemini_files:
        return None
    gemini_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return gemini_files[0]


def send_to_claude(topic: str, content: str) -> Path:
    """Write a report to Claude ensuring the 'gemini_' prefix is used."""
    slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in topic.lower()).strip("_")
    filename = f"gemini_{slug}.md"
    target = REPORTS_DIR / filename

    header = f"<!-- Gemini to Claude: {datetime.datetime.now().isoformat()} -->\n"
    full_content = header + content.strip() + "\n"

    target.write_text(full_content, encoding="utf-8")
    print(f"Delivered message to Claude at {target}")
    return target


def watch_for_claude_report(timeout_sec: int = 600) -> Path | None:
    """Wait for a new or updated claude_*.md report using inotifywait."""
    last_file = get_latest_claude_report()
    last_mtime = last_file.stat().st_mtime if last_file else 0

    print(f"Watching for Claude reports in {REPORTS_DIR} (last: {last_file.name if last_file else 'none'})...")

    cmd = [
        "inotifywait",
        "-q",
        "-e", "close_write,moved_to,create",
        "--timeout", str(timeout_sec),
        str(REPORTS_DIR)
    ]

    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Timeout or error
            break

        current_latest = get_latest_claude_report()
        if current_latest and current_latest.stat().st_mtime > last_mtime:
            print(f"New report from Claude detected: {current_latest.name}")
            return current_latest

    # Fallback check
    current_latest = get_latest_claude_report()
    if current_latest and current_latest.stat().st_mtime > last_mtime:
        return current_latest
    return None


def main():
    parser = argparse.ArgumentParser(description="Gemini-Claude Communication Bridge")
    subparsers = parser.add_subparsers(dest="command")

    # send
    send_parser = subparsers.add_parser("send", help="Send a message/report to Claude")
    send_parser.add_argument("--topic", required=True, help="Topic slug (e.g. review_of_ui)")
    send_parser.add_argument("--content", help="Markdown content string")
    send_parser.add_argument("--file", help="File to read content from")

    # watch
    watch_parser = subparsers.add_parser("watch", help="Wait for a report from Claude")
    watch_parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")

    # status
    subparsers.add_parser("status", help="Show latest reports from both Claude and Gemini")

    args = parser.parse_args()

    if args.command == "send":
        content = args.content or ""
        if args.file:
            content = Path(args.file).read_text(encoding="utf-8")
        if not content:
            content = sys.stdin.read()
        send_to_claude(args.topic, content)

    elif args.command == "watch":
        report = watch_for_claude_report(args.timeout)
        if report:
            print(f"LATEST_CLAUDE_REPORT: {report}")
            print("--- PREVIEW ---")
            lines = report.read_text(encoding="utf-8").splitlines()
            for line in lines[:25]:
                print(line)
            sys.exit(0)
        else:
            print("No new report from Claude received within timeout.")
            sys.exit(1)

    elif args.command == "status":
        latest_c = get_latest_claude_report()
        latest_g = get_latest_gemini_report()
        print("=== Claude & Gemini Bridge Status ===")
        print(f"Latest Claude report: {latest_c.name if latest_c else 'none'}")
        if latest_c:
            mtime = datetime.datetime.fromtimestamp(latest_c.stat().st_mtime)
            print(f"  Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latest Gemini report: {latest_g.name if latest_g else 'none'}")
        if latest_g:
            mtime = datetime.datetime.fromtimestamp(latest_g.stat().st_mtime)
            print(f"  Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
