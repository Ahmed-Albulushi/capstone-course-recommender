# ============================================================
# Pipeline 12 — Job Descriptions → TF-IDF (Filtered)
# ============================================================
# Query construction:
#   Career title → mapped to job query category →
#   concatenate all 120 job posting descriptions
#   for that category
#
# Retrieval:
#   TF-IDF vectorisation + cosine similarity
#   Library: scikit-learn (Pedregosa et al., 2011)
#   Vocabulary: full course corpus (3,424 courses)
#
# Re-ranking:
#   final_score = sim_score × (course_rating / 5.0)
#
# Domain filter: Applied post-scoring — NO fallback
#   Keep only courses tagged as:
#   computer-science, data-science, information-technology
#   Candidate pool: top 50 → filter → return all CS found
#   (may be fewer than 10 for some careers)
#
# Course text: Course Name + Course Description ONLY
#   Skills tags excluded — reserved for evaluation
#
# Input:
#   - datasets/cleaned/cs_students_excluded_careers.csv
#   - datasets/cleaned/Coursera_cleaned.csv
#   - datasets/cleaned/JobsDatasetProcessed.csv
#
# Output:
#   - results/recommendations/tfidf/filtered/p12_recommendations.csv
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
OUT  = os.path.join(BASE, 'results', 'recommendations', 'tfidf', 'filtered')

# ============================================================
#   Configuration
# ============================================================

TOP_N          = 10
TOP_CANDIDATES = 50
MAX_FEATS      = 20000

CS_DOMAINS = {
    'computer-science',
    'data-science',
    'information-technology',
}

KNOWN_DOMAINS = {
    'business', 'computer-science', 'data-science', 'life-sciences',
    'physical-science-and-engineering', 'social-sciences', 'arts-and-humanities',
    'information-technology', 'language-learning', 'personal-development', 'math-and-logic'
}

# ============================================================
#   Career → Job Query mapping (final agreed version)
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
}

# ============================================================
#   Step 1 — Load datasets
# ============================================================

students = pd.read_csv(os.path.join(DATA, 'cs_students_excluded_careers.csv'))
courses  = pd.read_csv(os.path.join(DATA, 'Coursera_cleaned.csv'))
jobs     = pd.read_csv(os.path.join(DATA, 'JobsDatasetProcessed.csv'), encoding='latin1')

# Normalise course rating
courses['Course Rating'] = pd.to_numeric(courses['Course Rating'], errors='coerce')
median_rating            = courses['Course Rating'].median()
courses['Course Rating'] = courses['Course Rating'].fillna(median_rating)
courses['rating_norm']   = courses['Course Rating'] / 5.0

# Extract broad domain from Skills tag
def extract_broad_domain(skills_str):
    if not isinstance(skills_str, str) or not skills_str.strip():
        return 'unknown'
    tokens = skills_str.strip().split()
    if len(tokens) < 2:
        return 'unknown'
    candidate = tokens[-2]
    return candidate if candidate in KNOWN_DOMAINS else 'unknown'

courses['broad_domain'] = courses['Skills'].apply(extract_broad_domain)
cs_count = courses['broad_domain'].isin(CS_DOMAINS).sum()

print('=' * 65)
print('PIPELINE 12 — Job Descriptions → TF-IDF (Filtered, No Fallback)')
print('=' * 65)
print(f'Students          : {len(students)}')
print(f'Courses           : {len(courses)}')
print(f'Job postings      : {len(jobs)}')
print(f'CS-domain courses : {cs_count}')
print(f'Top N             : {TOP_N}')
print(f'Candidate pool    : {TOP_CANDIDATES}')

# ============================================================
#   Step 2 — Build job description queries
# ============================================================

print('\nBuilding job description queries...')
query_descriptions = {}
for query_cat in jobs['Query'].unique():
    descriptions = jobs[jobs['Query'] == query_cat]['Description'].dropna().tolist()
    query_descriptions[query_cat] = ' '.join(descriptions)

print(f'Query categories built: {len(query_descriptions)}')

# ============================================================
#   Step 3 — Build course corpus
# ============================================================

course_texts = (
    courses['Course Name'].fillna('') + ' ' +
    courses['Course Description'].fillna('')
).tolist()

print(f'Course corpus built: {len(course_texts)} documents')

# ============================================================
#   Step 4 — TF-IDF matching + domain filter (no fallback)
# ============================================================

def recommend_tfidf_filtered(query, course_corpus, courses_df,
                              top_candidates=TOP_CANDIDATES, top_n=TOP_N):
    """
    Fits TF-IDF, scores all courses, fetches top_candidates.
    Filters to CS-domain only — no fallback.
    Returns however many CS courses are found (may be < top_n).
    """
    corpus  = [query] + course_corpus
    vec     = TfidfVectorizer(stop_words='english', max_features=MAX_FEATS)
    tfidf   = vec.fit_transform(corpus)
    scores  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_idx = scores.argsort()[-top_candidates:][::-1]

    candidates = [
        {
            'course':       courses_df.iloc[i]['Course Name'],
            'sim_score':    round(float(scores[i]), 4),
            'rating':       courses_df.iloc[i]['Course Rating'],
            'rating_norm':  round(courses_df.iloc[i]['rating_norm'], 4),
            'level':        courses_df.iloc[i]['Difficulty Level'],
            'skills':       courses_df.iloc[i]['Skills'],
            'broad_domain': courses_df.iloc[i]['broad_domain'],
        }
        for i in top_idx
    ]

    # Filter to CS-domain only — return all found, up to top_n
    cs_candidates = [c for c in candidates if c['broad_domain'] in CS_DOMAINS]
    return cs_candidates[:top_n]

def rerank(matches):
    for c in matches:
        c['final_score'] = round(c['sim_score'] * c['rating_norm'], 4)
    return sorted(matches, key=lambda x: x['final_score'], reverse=True)

# ============================================================
#   Step 5 — Run for all students (cached per career)
# ============================================================

all_recs     = []
skipped      = []
low_coverage = {}   # careers with fewer than TOP_N CS courses
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
        candidates           = recommend_tfidf_filtered(query, course_texts, courses)
        ranked               = rerank(candidates)
        career_cache[career] = ranked

        if len(ranked) < TOP_N:
            low_coverage[career] = len(ranked)
            print(f'  [INFO] {career} → [{query_cat}] — only {len(ranked)} CS courses found')
        else:
            print(f'  Computed: {career} → [{query_cat}]')

    ranked = career_cache.get(career)
    if not ranked:
        continue

    for rank, c in enumerate(ranked, 1):
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
            'broad_domain':      c['broad_domain'],
            'sim_score':         c['sim_score'],
            'final_score':       c['final_score'],
        })

# ============================================================
#   Step 6 — Save with metadata header
# ============================================================

os.makedirs(OUT, exist_ok=True)
out_path   = os.path.join(OUT, 'p12_recommendations.csv')
results_df = pd.DataFrame(all_recs)

with open(out_path, 'w') as f:
    f.write('# Pipeline      : P12 — Job Descriptions → TF-IDF (Filtered, No Fallback)\n')
    f.write('# Query         : Career title + concatenated job posting descriptions\n')
    f.write('# Retrieval     : TF-IDF cosine similarity\n')
    f.write('# Domain filter : computer-science, data-science, information-technology\n')
    f.write('# Fallback      : None — careers with <10 CS courses get fewer recs\n')
    f.write('# Students      : cs_students_excluded_careers.csv\n')
    f.write('# Courses       : Coursera_cleaned.csv\n')
    f.write(f'# Total recs    : {len(results_df)}\n')
    f.write(f'# Students cov  : {results_df["student_id"].nunique()}\n')
    f.write(f'# Careers cov   : {results_df["future_career"].nunique()}\n')
    f.write('#\n')
results_df.to_csv(out_path, mode='a', index=False)

print(f'\nTotal recommendations : {len(results_df)}')
print(f'Students covered      : {results_df["student_id"].nunique()}')
print(f'Careers covered       : {results_df["future_career"].nunique()}')
print(f'CS-domain rate        : {results_df["broad_domain"].isin(CS_DOMAINS).mean()*100:.1f}%')
print(f'Avg recs per student  : {len(results_df)/results_df["student_id"].nunique():.1f}')
if skipped:
    print(f'Skipped               : {set(skipped)}')
if low_coverage:
    print(f'\nCareers with < {TOP_N} CS courses found:')
    for career, count in sorted(low_coverage.items(), key=lambda x: x[1]):
        print(f'  {career}: {count} courses')
print(f'\nSaved → {out_path}')