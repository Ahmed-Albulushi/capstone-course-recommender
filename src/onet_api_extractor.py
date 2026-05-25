# ============================================================
# O*NET API Extractor — Enriched Career Data
# ============================================================
# Fetches richer occupation data from the O*NET Web Services
# API for all 17 retained careers:
#   - Occupation description
#   - Technology skills (all pages, including hot technologies)
#   - Tasks (all pages)
#
# This produces a richer query source than the basic O*NET
# xlsx description alone. Used as input to P1 and P2 pipelines
# instead of the description-only xlsx file.
#
# API: O*NET Web Services v2.0
#   Base: https://api-v2.onetcenter.org/online/occupations
#   Auth: X-API-Key header
#
# Input  : None (SOC codes hardcoded from O*NET xlsx)
# Output : datasets/cleaned/onet_enriched.csv
# ============================================================

import os
import time
import requests
import pandas as pd

BASE = '/Users/soesoe/Documents/Capstone Project/final_capstone-course-recommender'
DATA = os.path.join(BASE, 'datasets', 'cleaned')

API_KEY  = 'Xz0Un-j3zGa-mzRa3-bQU3O'
BASE_URL = 'https://api-v2.onetcenter.org/online/occupations'
HEADERS  = {'X-API-Key': API_KEY, 'Accept': 'application/json'}

# ============================================================
#   Career → SOC code mapping
#   SOC codes sourced directly from onet_occupation_data.xlsx
#   O*NET-SOC Code column — not manually guessed
# ============================================================

CAREER_SOC = {
    'Information Security Analyst': '15-1212.00',
    'Security Analyst':             '15-1212.00',
    'Machine Learning Researcher':  '15-1221.00',
    'AI Researcher':                '15-1221.00',
    'NLP Research Scientist':       '15-1221.00',
    'Cloud Solutions Architect':    '15-1241.00',
    'Database Administrator':       '15-1242.00',
    'Software Engineer':            '15-1252.00',
    'Web Developer':                '15-1254.00',
    'UX Designer':                  '15-1255.00',
    'Ethical Hacker':               '15-1299.04',
    'Digital Forensics Specialist': '15-1299.06',
    'Distributed Systems Engineer': '15-1299.08',
    'Data Scientist':               '15-2051.00',
    'Machine Learning Engineer':    '15-2051.00',
    'NLP Engineer':                 '15-2051.00',
    'Bioinformatician':             '19-1029.01',
}

# ============================================================
#   Fetch functions
# ============================================================

def get_description(soc_code):
    """Fetch occupation title and description."""
    r = requests.get(f"{BASE_URL}/{soc_code}", headers=HEADERS)
    if r.status_code == 200:
        data = r.json()
        return data.get('title', ''), data.get('description', '')
    print(f"  [WARN] Description fetch failed: {r.status_code}")
    return '', ''

def get_all_pages(url):
    """Fetch all paginated results from an endpoint."""
    items = []
    while url:
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"  [WARN] Page fetch failed: {r.status_code} — {url}")
            break
        data = r.json()
        # Tasks
        if 'task' in data:
            items.extend([t['title'] for t in data.get('task', [])])
        # Technology skills — category titles + specific examples
        if 'category' in data:
            for cat in data.get('category', []):
                items.append(cat['title'])
                for ex in cat.get('example', []):
                    items.append(ex['title'])
        url = data.get('next')
        time.sleep(0.2)  # polite rate limiting
    return items

def fetch_career_data(soc_code):
    """Fetch description + tech skills + tasks for one SOC code."""
    title, description = get_description(soc_code)
    time.sleep(0.3)

    tech_skills = get_all_pages(f"{BASE_URL}/{soc_code}/summary/technology_skills")
    time.sleep(0.3)

    tasks = get_all_pages(f"{BASE_URL}/{soc_code}/summary/tasks")
    time.sleep(0.3)

    return {
        'onet_title':   title,
        'description':  description,
        'tech_skills':  ' | '.join(tech_skills),
        'tasks':        ' | '.join(tasks),
        'tech_count':   len(tech_skills),
        'task_count':   len(tasks),
    }

# ============================================================
#   Main — fetch all careers
#   Cache by SOC code to avoid duplicate API calls
# ============================================================

print('=' * 65)
print('O*NET API EXTRACTOR — Enriched Career Data')
print('=' * 65)
print(f'Careers  : {len(CAREER_SOC)}')
print(f'Unique SOC codes: {len(set(CAREER_SOC.values()))}')

soc_cache = {}
rows      = []

for career, soc_code in CAREER_SOC.items():
    print(f'\nFetching: {career} ({soc_code})')

    if soc_code not in soc_cache:
        data = fetch_career_data(soc_code)
        soc_cache[soc_code] = data
        print(f'  Title      : {data["onet_title"]}')
        print(f'  Tech skills: {data["tech_count"]} items')
        print(f'  Tasks      : {data["task_count"]} items')
        print(f'  Desc chars : {len(data["description"])}')
    else:
        print(f'  [CACHED] Using data from previous fetch')
        data = soc_cache[soc_code]

    rows.append({
        'career':      career,
        'soc_code':    soc_code,
        'onet_title':  data['onet_title'],
        'description': data['description'],
        'tech_skills': data['tech_skills'],
        'tasks':       data['tasks'],
        'tech_count':  data['tech_count'],
        'task_count':  data['task_count'],
    })

# ============================================================
#   Save
# ============================================================

out_path = os.path.join(DATA, 'onet_enriched.csv')
df       = pd.DataFrame(rows)
df.to_csv(out_path, index=False)

print(f'\n{"="*65}')
print(f'COMPLETE')
print(f'{"="*65}')
print(f'Careers saved : {len(df)}')
print(f'Saved → {out_path}')
print(f'\nColumns: career | soc_code | onet_title | description | tech_skills | tasks')
print(f'\nSample tech_skills (Data Scientist):')
ds = df[df['career']=='Data Scientist'].iloc[0]
print(f"  {ds['tech_skills'][:200]}...")
print(f'\nSample tasks (Data Scientist):')
print(f"  {ds['tasks'][:200]}...")