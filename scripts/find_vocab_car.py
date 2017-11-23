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

            if turn['turn'] == 'assistant':
                r = turn['data']['utterance']
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
        kbs.append(dialogue['scenario']['kb'])

    print 'Loaded %i navigate data!'%len(navigate_data)
    print 'Loaded %i schedule data!'%len(schedule_data)
    print 'Loaded %i weather data!'%len(weather_data)

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
    df.to_csv('datasets/incar_addr_poi.csv')


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


def dump_vocab():
    lookup = pd.read_csv('datasets/incar_addr_poi.csv')

    vocab = set()
    specials = ['100.', '20.']
    for key in specials:
        vocab.add(key)

    for file in files:
        print 'Now processing ' + file
        data = load_json_dataset(file)
        navigates, weathers, schedules, _ = load_incar_data(data)

        for navigate in navigates:
            for u, r in navigate:
                if u:
                    u = look4str(u, lookup)
                    for token in word_tokenize(u):
                        vocab.add(token.lower())

                r = look4str(r, lookup)
                for token in word_tokenize(r):
                    vocab.add(token.lower())

        print 'Finished navigations... len(vocab) = ', len(vocab)

        for weather in weathers:
            for u, r in weather:
                if u:
                    for token in word_tokenize(u.lower()):
                        vocab.add(token.lower())
                for token in word_tokenize(r):
                    vocab.add(token.lower())

        print 'Finished weathers... len(vocab) = ', len(vocab)

        for schedule in schedules:
            for u, r in schedule:
                if u:
                    for token in word_tokenize(u):
                        vocab.add(token.lower())
                for token in word_tokenize(r):
                    vocab.add(token.lower())


        print 'Done with ' + file + 'now, the vocab has %i words...'%len(vocab)

    vocab = list(vocab)
    vocab.sort()
    special_tokens = ["<SILENCE>", "<T01>","<T02>","<T03>","<T04>","<T05>","<T06>",
                      "<T07>","<T08>","<T09>","<T10>","<T11>","<T12>","<T13>",
                      "<T14>","UNK", "SOS", "EOS", "api_call","poi", "addr"]
    all_tokens = special_tokens + vocab

    print len(all_tokens)

    json.dump(all_tokens, open("car_vocab.json", "w"))

dump_vocab()