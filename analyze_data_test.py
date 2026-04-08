"""
Simple analysis of the data_test dataset.
Reads journal entries and calculates mood statistics.
"""

import json
import os

DATA_PATH = os.path.join("data_test", "psychoanalytical.json")


def load_entries(path):
    with open(path, "r") as f:
        return json.load(f)


def calculate_stats(entries):
    moods = [e["mood"] for e in entries]

    total = len(moods)
    average = sum(moods) / total
    highest = max(moods)
    lowest = min(moods)
    good_days = sum(1 for m in moods if m >= 6)

    return {
        "total_entries": total,
        "average_mood": round(average, 2),
        "highest_mood": highest,
        "lowest_mood": lowest,
        "good_days": good_days,
        "good_days_percent": round((good_days / total) * 100, 1),
    }


def find_best_and_worst(entries):
    best = max(entries, key=lambda e: e["mood"])
    worst = min(entries, key=lambda e: e["mood"])
    return best, worst


def main():
    entries = load_entries(DATA_PATH)
    stats = calculate_stats(entries)
    best, worst = find_best_and_worst(entries)

    print("=== Journal Mood Analysis ===\n")
    print(f"Total entries:     {stats['total_entries']}")
    print(f"Average mood:      {stats['average_mood']} / 10")
    print(f"Highest mood:      {stats['highest_mood']} / 10")
    print(f"Lowest mood:       {stats['lowest_mood']} / 10")
    print(f"Good days (≥6):    {stats['good_days']} ({stats['good_days_percent']}%)")

    print("\n--- Best Day ---")
    print(f"  Date:  {best['date']}")
    print(f"  Title: {best['title']}")
    print(f"  Mood:  {best['mood']}")

    print("\n--- Worst Day ---")
    print(f"  Date:  {worst['date']}")
    print(f"  Title: {worst['title']}")
    print(f"  Mood:  {worst['mood']}")


if __name__ == "__main__":
    main()
