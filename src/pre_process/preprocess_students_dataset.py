# ============================================================
# Pre-processing — Student Dataset Cleaning
# ============================================================
# Produces cs_students_excluded_careers.csv used by all
# recommendation pipelines and the evaluation script.
#
# Exclusion criteria — three categories:
#
#   1. No O*NET occupation exists:
#      Quantum Computing Researcher, VR Developer,
#      Robotics Engineer, Blockchain Engineer, SEO Specialist
#
#   2. Emerging roles not yet in O*NET taxonomy:
#      Mobile App Developer, Computer Vision Engineer,
#      DevOps Engineer, IoT Developer, Embedded Software Engineer
#
#   3. O*NET row found but description is wrong/misleading:
#      Data Privacy Specialist  (row 134 = Information Security Engineers)
#      Healthcare IT Specialist (row 119 = Database Admins, no healthcare)
#      Graphics Programmer      (row 123 = generic Computer Programmers)
#
# Justification: exclusion ensures fairness — only careers
# with a valid, semantically aligned O*NET description are
# included, maintaining consistency across all pipelines.
#
# Input  : datasets/cleaned/cs_students_cleaned.csv
# Output : datasets/cleaned/cs_students_excluded_careers.csv
# ============================================================

import os
import pandas as pd

BASE     = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA     = os.path.join(BASE, 'datasets', 'cleaned')
IN_FILE  = os.path.join(DATA, 'cs_students_cleaned.csv')
OUT_FILE = os.path.join(DATA, 'cs_students_excluded_careers.csv')

EXCLUDED_CAREERS = {
    # Category 1 — No O*NET occupation exists
    'Quantum Computing Researcher',
    'VR Developer',
    'Robotics Engineer',
    'Blockchain Engineer',
    'SEO Specialist',
    # Category 2 — Emerging roles not in O*NET taxonomy
    'Mobile App Developer',
    'Computer Vision Engineer',
    'DevOps Engineer',
    'IoT Developer',
    'Embedded Software Engineer',
    # Category 3 — Wrong/misleading O*NET row
    'Data Privacy Specialist',
    'Healthcare IT Specialist',
    'Graphics Programmer',
    # Category 4 — No O*NET occupation with sufficient alignment
    'Data Analyst',
    'Game Developer',
}

students      = pd.read_csv(IN_FILE)
excluded      = students[students['Future Career'].isin(EXCLUDED_CAREERS)]
students_eval = students[~students['Future Career'].isin(EXCLUDED_CAREERS)].reset_index(drop=True)

print('=' * 65)
print('STUDENT DATASET PRE-PROCESSING')
print('=' * 65)
print(f'Original students     : {len(students)}')
print(f'Excluded careers      : {len(EXCLUDED_CAREERS)}')
print(f'Excluded students     : {len(excluded)} ({len(excluded)/len(students)*100:.1f}%)')
print(f'\nBreakdown:')
print(excluded['Future Career'].value_counts().to_string())
print(f'\nRetained students     : {len(students_eval)}')
print(f'Retained careers      : {students_eval["Future Career"].nunique()}')

students_eval.to_csv(OUT_FILE, index=False)
print(f'\nSaved → {OUT_FILE}')
print('\nCareer distribution:')
print(students_eval['Future Career'].value_counts().to_string())