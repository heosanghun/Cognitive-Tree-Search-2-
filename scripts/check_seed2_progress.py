import sys
from pathlib import Path

def main():
    log_path = Path("results/post_s2_autopilot/logs/wave2.log")
    if not log_path.is_file():
        print("Log file not found.")
        return 1
    
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()
        
    idx = log_content.rfind("math500 seed=2")
    if idx == -1:
        print("Active seed 2 not started in log yet.")
        return 0
        
    sub = log_content[idx:]
    count = sub.count("sc_14/math500")
    print(f"Active Seed 2 Progress: {count} / 50 problems evaluated")
    return 0

if __name__ == "__main__":
    sys.exit(main())
