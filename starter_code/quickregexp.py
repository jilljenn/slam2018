from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import re


DATASET = 'es_en'
countries = defaultdict(set)
outcomes = defaultdict(lambda: defaultdict(list))
count = Counter()
who = re.compile('# user:([^ ]+)  countries:([^ ]+)  days:([^ ]+)  client:([^ ]+)  session:([^ ]+)  format:([^ ]+)  time:([^ ]+)')
what = re.compile('[^ ]{12}  ([^ ]+).*(0|1)')
rows = []


with open(f'data_{DATASET}/{DATASET}.slam.20190204.train') as f:
    for line in f.read().splitlines():
        if not line:
            continue
        m = who.match(line)
        if m:
            user_id, countries, days, client, session, exercise_type, duration = m.groups()
            if '|' in countries:
                countries = countries.split('|')[0]  # Keep first
            continue
        m = what.match(line)
        if m:
            token, inv_outcome = m.groups()
            outcome = 1 - int(inv_outcome)
            outcomes[countries][token].append(outcome)
            rows.append({
                'user_id': user_id,
                'token': token,
                'correct': outcome,
                'country': countries,
                'days': days,
                'client': client,
                'session': session,
                'exercise_type': exercise_type,
                'duration': duration,
                'wins': count[user_id, token, 1],
                'fails': count[user_id, token, 0]
            })
            count[user_id, token, outcome] += 1
            continue
        if 'prompt' in line:
            continue
        print('Could not', line)
        break


df = pd.DataFrame.from_dict(rows)
df['user'] = np.unique(df['user_id'], return_inverse=True)[1]
df['skill'] = np.unique(df['token'], return_inverse=True)[1]
print(df.head())
df.to_csv(f'~/code/ktm/data/{DATASET}/data.csv', index=None)

'''     for user_id, country in r.findall(f.read()):
            countries[user_id].add(country)'''

# print(Counter(map(' '.join, countries.values())))
