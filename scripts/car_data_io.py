import json
from nltk import word_tokenize
import pandas as pd

def load_json_dataset(path):
    '''
    Load the in-car dataset as it is
    :param path: path to train/validation/test.json
    :return: the json file
    '''
    with open(path) as f:
        data = json.load(f)
    print path + ' file loaded!!'
    return data


def look4str(u, df):
    a = df['addrs'].apply(lambda x: x in u)
    b = df['pois'].apply(lambda x: x in u)
    a = df[a]['addrs'].as_matrix()
    b = df[b]['pois'].as_matrix()

    if len(a) != 0:
        u = u.replace(a[0], 'addr')
    if len(b) != 0:
        u = u.replace(b[0], 'poi')
    return u


def load_incar_data(data_json):
    '''
    :param data_json: a json file loaded from .json
    :return:
    navigate/weather/schedule_data, three lists for the three tasks
    each list is a list of dialogues, and each dialogue is a list of turns [(u1, r1), (u2, r2)...]
    each utterance/response is a list of tokens
    '''
    lookup = pd.read_csv('datasets/incar_addr_poi.csv')
    navigate_data = []
    schedule_data = []
    weather_data = []
    kbs = []

    uu = []
    rr = []

    for dialogue in data_json:
        dia = []
        u = None
        r = None
        for turn in dialogue['dialogue']:
            if turn['turn'] == 'driver':
                u = turn['data']['utterance']
                u = look4str(u, lookup) # Comment out this line to get raw dataset without replacement
                u = word_tokenize(u.lower())

                uu.append(len(u))
            if turn['turn'] == 'assistant':
                r = turn['data']['utterance']
                r = look4str(r, lookup) # Comment out this line to get raw dataset without replacement
                r = word_tokenize(r.lower())

                rr.append(len(r))
                if len(r) == 95:
                    print r
                dia.append((u, r))

        if dialogue['scenario']['task']['intent'] == 'navigate':
            navigate_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'schedule':
            schedule_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'weather':
            weather_data.append(dia)
        else:
            print dialogue['scenario']['task']['intent']

        kbs.append(dialogue['scenario']['kb'])
    print 'Loaded %i navigate data!'%len(navigate_data)
    print 'Loaded %i schedule data!'%len(schedule_data)
    print 'Loaded %i weather data!'%len(weather_data)

    print max(uu)
    print max(rr)

    return navigate_data, weather_data, schedule_data, kbs

files = ['datasets/in_car/train.json','datasets/in_car/dev.json',
         'datasets/in_car/test.json']

for file in files:
    data = load_json_dataset(file)
    navigates, weathers, schedules, _ = load_incar_data(data)