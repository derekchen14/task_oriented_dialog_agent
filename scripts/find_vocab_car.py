from nltk import word_tokenize
import json
import pandas as pd

files = ['datasets/in_car/train.json','datasets/in_car/dev.json',
         'datasets/in_car/test.json']
entity = 'datasets/in_car/entities.json'


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


def load_incar_data(data_json):
    '''
    :param data_json: a json file loaded from .json
    :return:
    navigate/weather/schedule_data, three lists for the three tasks
    each list is a list of dialogues, and each dialogue is a list of turns [(u1, r1), (u2, r2)...]
    each utterance/response is a list of tokens
    '''
    navigate_data = []
    schedule_data = []
    weather_data = []
    kbs = []

    for dialogue in data_json:
        dia = []
        u = None
        r = None
        for turn in dialogue['dialogue']:
            if turn['turn'] == 'driver':
                u = turn['data']['utterance']
                # u = word_tokenize(u)
            if turn['turn'] == 'assistant':
                r = turn['data']['utterance']
                # r = word_tokenize(r)

                dia.append((u, r))
                u = None
                r = None

        if dialogue['scenario']['task']['intent'] == 'navigate':
            navigate_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'schedule':
            schedule_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'weather':
            weather_data.append(dia)
        else:
            print dialogue['scenario']['task']['intent']

        print 'Loaded %i navigate data!'%len(navigate_data)
        print 'Loaded %i schedule data!'%len(schedule_data)
        print 'Loaded %i weather data!'%len(weather_data)

        kbs.append(dialogue['scenario']['kb'])

    return navigate_data, weather_data, schedule_data, kbs


# Step 1. Get addresses and pois
def save_addr_pois():
    addrs = set()
    poiSet = set()

    data = load_json_dataset(entity)
    pois = data['poi']

    for poi in pois:
        addrs.add(poi['address'])
        poiSet.add(poi['poi'])

    df = pd.DataFrame(data={'addrs':list(addrs), 'pois':list(poiSet)})
    df.to_csv('datasets/incar_addr_poi.csv', index=False)


# Step 2.
