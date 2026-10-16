
from pycaret.classification import (
    setup,
    compare_models,
    finalize_model
)
from pycaret.datasets import get_data
import joblib

# Load the Iris dataset
iris = get_data("iris")

print("First five rows:")
print(iris.head())

# Configure the classification experiment
setup(
    data=iris,
    target="species",
    session_id=42,
    verbose=False,
    html=False,
    log_experiment=False
)

# Compare available classification algorithms
best_model = compare_models()

print("\nBest model selected by PyCaret:")
print(best_model)

# Train the selected model on the full dataset
final_model = finalize_model(best_model)

# Save the trained model
joblib.dump(final_model, "iris_model.pkl")

print("\nModel saved as iris_model.pkl")
