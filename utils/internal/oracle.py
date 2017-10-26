import numpy as np
np.random.seed(11)

def pause():
    programPause = raw_input()

def get_oracle_dialogs(lines, n=20):
    '''
    :param lines: f.lines()
    :param n: number of dialogs to load
    :return: [[(u1, r1), (u2, r2)...], [(u1, r1), (u2, r2)...], ...]
             ui and ri are strings
    '''
    data = []
    dialogue = []
    kb_dialogue = []
    kb = []
    read_dialog = 0
    for line in lines:
        if read_dialog == n:
            break

        if line != '\n' and line != lines[-1]:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            line = line.decode('utf-8').strip()


            if len(line.split('\t')) == 1:
                kb_dialogue += (line.split('\t'))
                continue

            q, a = line.split('\t')
            dialogue.append((q, a))
        else:
            data.append(dialogue)
            kb.append(kb_dialogue)
            read_dialog += 1
            dialogue = []
            kb_dialogue = []
    return data, kb


def oracleExperiment(files, num_dialog_each_file, directory='datasets/restaurants/'):
    '''
    :param files: list of files you want to read from
    :param num_dialog_each_file: int
    :param directory: data directory
    :return:
    '''
    order = np.random.permutation(num_dialog_each_file * len(files))
    data = []
    kbs = []
    for file in files:
        path = directory + file
        with open(path) as f:
            datus, kb = get_oracle_dialogs(f.readlines(), n=num_dialog_each_file)
            data += datus
            kbs += kb

    for i in order:
        print "--------NEW CUSTOMER--------"
        for turn in data[i]:
            print turn[0]
            pause()


directory = 'datasets/restaurants/'
files = ['dialog-babi-task5-full-dialogs-trn.txt', 'dialog-babi-task6-dstc2-trn.txt']
num_dialog_each_file = 2

oracleExperiment(files, num_dialog_each_file)





