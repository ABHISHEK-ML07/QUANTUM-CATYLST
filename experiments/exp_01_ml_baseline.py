# experiments/exp_01_ml_baseline.py
"""
Run ML baseline experiment: trains RandomForest on DFT subset and saves artifacts.
"""

from src.ml_model import train_and_evaluate

def main():
    print("Running ML baseline experiment...")
    res = train_and_evaluate(random_state=123)
    print("Experiment finished. Summary:")
    print(res)

if __name__ == "__main__":
    main()