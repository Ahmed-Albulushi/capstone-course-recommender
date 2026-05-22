# ============================================================
# Pipeline 9 — Job Descriptions → TF-IDF (Unfiltered)
# ============================================================
# Query construction:
#   Career title → mapped to job query category →
#   concatenate all 120 job posting descriptions
#   for that category (~460K chars total per career)
#
# Retrieval:
#   TF-IDF vectorisation + cosine similarity
#   Library: scikit-learn (Pedregosa et al., 2011)
#   Vocabulary: full course corpus (3,424 courses)
#
# Re-ranking:
#   final_score = sim_score × (course_rating / 5.0)
#
# Course text: Course Name + Course Description ONLY
#   Skills tags excluded — reserved for evaluation
#
# Domain filter: None
#
# Input:
#   - datasets/cleaned/cs_students_cleaned.csv
#   - datasets/cleaned/Coursera_cleaned.csv
#   - datasets/cleaned/JobsDatasetProcessed.csv
#
# Output:
#   - results/recommendations/tfidf/unfiltered/p9_recommendations.csv
# ============================================================

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
#   Paths
# ============================================================

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA = os.path.join(BASE, 'datasets', 'cleaned')
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf', 'unfiltered')

# ============================================================
#   Configuration
# ============================================================

TOP_N     = 10
MAX_FEATS = 20000

# ============================================================
#   Career → Job Query mapping
#   Maps each student career to the closest job posting category
#   (25 categories in JobsDatasetProcessed.csv)
# ============================================================

CAREER_TO_QUERY = {
    # --- Exact matches ---
    'AI Researcher':                'Artificial Intelligence',
    'Machine Learning Researcher':  'Machine Learning',
    'Machine Learning Engineer':    'Machine Learning',
    'Computer Vision Engineer':     'Deep Learning',
    'Data Scientist':               'Data Scientist',
    'Data Analyst':                 'Data Analyst',
    'Database Administrator':       'Database Administrator',
    'Cloud Solutions Architect':    'Cloud Architect',
    'Web Developer':                'Full Stack Developer',
    'Mobile App Developer':         'Full Stack Developer',
    'Information Security Analyst': 'Information Security Analyst',
    'Security Analyst':             'Information Security Analyst',
    'Ethical Hacker':               'Information Security Analyst',
    'Digital Forensics Specialist': 'Information Security Analyst',
    'Data Privacy Specialist':      'Information Security Analyst',
    'DevOps Engineer':              'Technical Operations',
    'IoT Developer':                'Technology Integration',
    'Network Architect':            'Network Architect',
    # --- Expert mapping improvements ---
    'NLP Research Scientist':       'Deep Learning',
    'NLP Engineer':                 'Deep Learning',
    'Software Engineer':            'Technology Integration',
    'Embedded Software Engineer':   'Technical Operations',
    'Healthcare IT Specialist':     'IT Consultant',
    'Bioinformatician':             'Data Scientist',
    # --- Our mapping retained ---
    'Distributed Systems Engineer': 'Cloud Services Developer',
    'Graphics Programmer':          'Data Visualization Expert',
    'Game Developer':               'Full Stack Developer',
    'UX Designer':                  'Full Stack Developer',
    'Geospatial Analyst':           'Data Visualization Expert',
    # --- Excluded from evaluation (no reliable job query match)
    #     Recommendations still generated — eval only exclusion
    # 'Quantum Computing Researcher': excluded
    # 'VR Developer':                 excluded
    # 'Robotics Engineer':            excluded
    # 'Blockchain Engineer':          excluded
    # 'SEO Specialist':               excluded
}

# ============================================================
#   Step 1 — Load datasets
# ============================================================

students = pd.read_csv(os.path.join(DATA, 'cs_students_excluded_careers.csv'))
courses  = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))
jobs     = pd.read_csv(os.path.join(DATA, 'JobsDatasetProcessed.csv'), encoding='latin1')

# Normalise course rating: replace non-numeric with median, scale to [0, 1]
courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
median_rating            = courses['Course Rating'].median()
courses['Course Rating'] = courses['Course Rating'].fillna(median_rating)
courses['rating_norm']   = courses['Course Rating'] / 5.0

print('=' * 65)
print('PIPELINE 9 — Job Descriptions → TF-IDF (Unfiltered)')
print('=' * 65)
print(f'Students     : {len(students)}')
print(f'Courses      : {len(courses)}')
print(f'Job postings : {len(jobs)}')
print(f'Top N        : {TOP_N}')

# ============================================================
#   Step 2 — Build job description query per career
#            Concatenate all 120 descriptions per query category
# ============================================================

print('\nBuilding job description queries...')
query_descriptions = {}
for query_cat in jobs['Query'].unique():
    descriptions = jobs[jobs['Query'] == query_cat]['Description'].dropna().tolist()
    query_descriptions[query_cat] = ' '.join(descriptions)

print(f'Query categories built: {len(query_descriptions)}')

# ============================================================
#   Step 3 — Build course corpus
#            Course Name + Description only
#            Skills tags excluded for evaluation independence
# ============================================================

course_texts = (
    courses['Course Name'].fillna('') + ' ' +
    courses['Course Description'].fillna('')
).tolist()

print(f'Course corpus built: {len(course_texts)} documents')

# ============================================================
#   Step 4 — TF-IDF matching function
# ============================================================

def recommend_tfidf(query, course_corpus, courses_df, top_n=TOP_N):
    """
    Fits TF-IDF on [query] + full course corpus.
    Returns top_n courses ranked by cosine similarity.
    """
    corpus  = [query] + course_corpus
    vec     = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf   = vec.fit_transform(corpus)
    scores  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]

    return [
        {
            'course':      courses_df.iloc[i]['Course Name'],
            'sim_score':   round(float(scores[i]), 4),
            'rating':      courses_df.iloc[i]['Course Rating'],
            'rating_norm': round(courses_df.iloc[i]['rating_norm'], 4),
            'level':       courses_df.iloc[i]['Difficulty Level'],
            'skills':      courses_df.iloc[i]['Skills'],
        }
        for i in top_idx
    ]

def rerank(matches):
    for c in matches:
        c['final_score'] = round(c['sim_score'] * c['rating_norm'], 4)
    return sorted(matches, key=lambda x: x['final_score'], reverse=True)

# ============================================================
#   Step 5 — Run for all students
#            TF-IDF computed once per unique career (cached)
# ============================================================

all_recs     = []
skipped      = []
career_cache = {}

print('\nGenerating recommendations...')

for _, student in students.iterrows():
    career = student['Future Career']

    if career not in career_cache:
        query_cat = CAREER_TO_QUERY.get(career)

        if not query_cat or query_cat not in query_descriptions:
            print(f'  [SKIP] No job query mapping for: {career}')
            skipped.append(career)
            career_cache[career] = None
            continue

        query                = career + ' ' + query_descriptions[query_cat]
        ranked               = rerank(recommend_tfidf(query, course_texts, courses))
        career_cache[career] = ranked
        print(f'  Computed: {career} → [{query_cat}]')

    ranked = career_cache.get(career)
    if not ranked:
        continue

    for rank, c in enumerate(ranked[:TOP_N], 1):
        all_recs.append({
            'student_id':        student['Student ID'],
            'student_name':      student['Name'],
            'future_career':     career,
            'job_query':         CAREER_TO_QUERY.get(career, ''),
            'interested_domain': student['Interested Domain'],
            'rank':              rank,
            'course':            c['course'],
            'rating':            c['rating'],
            'rating_norm':       c['rating_norm'],
            'level':             c['level'],
            'skills':            c['skills'],
            'sim_score':         c['sim_score'],
            'final_score':       c['final_score'],
        })

# ============================================================
#   Step 6 — Save results
# ============================================================

os.makedirs(OUT, exist_ok=True)
out_path   = os.path.join(OUT, 'p9_recommendations.csv')
results_df = pd.DataFrame(all_recs)
results_df.to_csv(out_path, index=False)

print(f'\nTotal recommendations : {len(results_df)}')
print(f'Students covered      : {results_df["student_id"].nunique()}')
print(f'Careers covered       : {results_df["future_career"].nunique()}')
if skipped:
    print(f'Skipped               : {set(skipped)}')
print(f'\nSaved → {out_path}')