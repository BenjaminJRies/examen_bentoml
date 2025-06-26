import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student admissions dataset
n_samples = 400

# Generate features
gre_scores = np.random.normal(315, 12, n_samples)
gre_scores = np.clip(gre_scores, 260, 340)

toefl_scores = np.random.normal(107, 6, n_samples)
toefl_scores = np.clip(toefl_scores, 92, 120)

university_rating = np.random.randint(1, 6, n_samples)
sop = np.random.uniform(1, 5, n_samples)
lor = np.random.uniform(1, 5, n_samples)
cgpa = np.random.normal(8.5, 0.6, n_samples)
cgpa = np.clip(cgpa, 6.8, 10.0)

research = np.random.binomial(1, 0.55, n_samples)

# Calculate chance of admit based on realistic relationships
chance_of_admit = (
    0.002 * gre_scores +
    0.004 * toefl_scores +
    0.06 * university_rating +
    0.08 * sop +
    0.07 * lor +
    0.12 * cgpa +
    0.08 * research +
    np.random.normal(0, 0.05, n_samples) -
    2.2
)

# Normalize chance of admit to be between 0 and 1
chance_of_admit = np.clip(chance_of_admit, 0, 1)

# Create DataFrame
data = pd.DataFrame({
    'GRE Score': gre_scores.round().astype(int),
    'TOEFL Score': toefl_scores.round().astype(int),
    'University Rating': university_rating,
    'SOP': sop.round(1),
    'LOR': lor.round(1),
    'CGPA': cgpa.round(2),
    'Research': research,
    'Chance of Admit': chance_of_admit.round(3)
})

# Save the dataset
data.to_csv('/home/ubuntu/examen_bentoml/data/raw/admission_data.csv', index=False)

print("Dataset created successfully!")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset statistics:")
print(data.describe())
