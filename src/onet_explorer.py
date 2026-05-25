# ============================================================
# O*NET API Explorer v2 — Full data fetch
# ============================================================
# Fetches for each career:
#   - Description (from occupation endpoint)
#   - Technology skills (all pages)
#   - Tasks (all pages)
#
# Also tests no-match careers using keyword search
# to see if O*NET has any relevant occupation
# ============================================================

import requests
import json

API_KEY  = "Xz0Un-j3zGa-mzRa3-bQU3O"
BASE_URL = "https://api-v2.onetcenter.org"
HEADERS  = {"X-API-Key": API_KEY, "Accept": "application/json"}

def get_all_pages(url):
    """Fetch all paginated results from an endpoint."""
    results = []
    while url:
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            break
        data = r.json()
        # Tasks
        if 'task' in data:
            results.extend([t['title'] for t in data['task']])
        # Technology skills
        if 'category' in data:
            for cat in data['category']:
                results.append(cat['title'])
                if 'example' in cat:
                    results.extend([e['title'] for e in cat['example']])
        url = data.get('next')
    return results

def get_occupation(soc_code):
    """Get basic occupation info including description."""
    r = requests.get(f"{BASE_URL}/online/occupations/{soc_code}", headers=HEADERS)
    if r.status_code == 200:
        return r.json()
    return None

def search_career(keyword):
    """Search O*NET for a career by keyword."""
    r = requests.get(
        f"{BASE_URL}/online/search",
        headers=HEADERS,
        params={"keyword": keyword, "start": 1, "end": 5}
    )
    if r.status_code == 200:
        return r.json()
    return None

# ── Test full fetch for Data Scientists ──
print("=" * 60)
print("FULL FETCH — Data Scientists (15-2051.00)")
print("=" * 60)

soc = "15-2051.00"
occ = get_occupation(soc)
if occ:
    print(f"\nTitle      : {occ.get('title')}")
    print(f"Description: {occ.get('description', '')[:300]}...")

tech = get_all_pages(f"{BASE_URL}/online/occupations/{soc}/summary/technology_skills")
print(f"\nTechnology Skills ({len(tech)} items):")
for t in tech[:20]:
    print(f"  - {t}")

tasks = get_all_pages(f"{BASE_URL}/online/occupations/{soc}/summary/tasks")
print(f"\nTasks ({len(tasks)} items):")
for t in tasks[:10]:
    print(f"  - {t}")

# ── Search for no-match careers ──
print("\n\n" + "=" * 60)
print("KEYWORD SEARCH — No-match careers")
print("=" * 60)

no_match_careers = [
    'Mobile App Developer',
    'DevOps Engineer',
    'Computer Vision Engineer',
    'IoT Developer',
    'Embedded Software Engineer',
    'Data Privacy Specialist',
    'Healthcare IT Specialist',
    'Graphics Programmer',
]

for career in no_match_careers:
    print(f"\n{career}:")
    results = search_career(career)
    if results and 'occupation' in results:
        for occ in results['occupation'][:3]:
            print(f"  [{occ['code']}] {occ['title']}")
    else:
        print("  No results found")