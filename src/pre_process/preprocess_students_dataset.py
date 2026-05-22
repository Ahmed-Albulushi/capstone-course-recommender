# ============================================================
# Pre-processing — Student Dataset Cleaning
# ============================================================
# Purpose:
#   Produces a cleaned student dataset for use across all
#   recommendation pipelines and evaluation scripts.
#
# Exclusions:
#   Five career categories are excluded because the job
#   postings dataset contains no query category with
#   sufficient semantic overlap to serve as a reliable
#   ground truth source:
#
#     - Quantum Computing Researcher
#     - VR Developer
#     - Robotics Engineer
#     - Blockchain Engineer
#     - SEO Specialist
#
#   These careers represent emerging or specialised domains
#   underrepresented in the 25-category job posting taxonomy.
#   Their exclusion affects 8 of 180 students (4.4%) and
#   does not materially alter the evaluation scope.
#
# Input:
#   - datasets/cleaned/cs_students_cleaned.csv   (180 students)
#
# Output:
#   - datasets/cleaned/cs_students_eval.csv      (172 students)
# ============================================================

import os
import pandas as pd

# ============================================================
#   Paths
# ============================================================

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA = os.path.join(BASE, 'datasets', 'cleaned')

IN_FILE  = os.path.join(DATA, 'cs_students_cleaned.csv')
OUT_FILE = os.path.join(DATA, 'cs_students_exluded_careers.csv')

# ============================================================
#   Excluded careers
#   (no reliable job query match in job postings dataset)
# ============================================================

EXCLUDED_CAREERS = {
    'Quantum Computing Researcher',
    'VR Developer',
    'Robotics Engineer',
    'Blockchain Engineer',
    'SEO Specialist',
}

# ============================================================
#   Load and filter
# ============================================================

students = pd.read_csv(IN_FILE)

print('=' * 65)
print('STUDENT DATASET PRE-PROCESSING')
print('=' * 65)
print(f'Original students     : {len(students)}')
print(f'Unique careers        : {students["Future Career"].nunique()}')

# Show excluded students
excluded = students[students['Future Career'].isin(EXCLUDED_CAREERS)]
print(f'\nExcluded careers      : {sorted(EXCLUDED_CAREERS)}')
print(f'Excluded students     : {len(excluded)} ({len(excluded)/len(students)*100:.1f}%)')
print('\nBreakdown:')
print(excluded['Future Career'].value_counts().to_string())

# Filter
students_eval = students[~students['Future Career'].isin(EXCLUDED_CAREERS)].copy()
students_eval = students_eval.reset_index(drop=True)

print(f'\nRetained students     : {len(students_eval)}')
print(f'Retained careers      : {students_eval["Future Career"].nunique()}')

# ============================================================
#   Save
# ============================================================

students_eval.to_csv(OUT_FILE, index=False)

print(f'\nSaved → {OUT_FILE}')
print('\nCareer distribution in eval dataset:')
print(students_eval['Future Career'].value_counts().to_string())