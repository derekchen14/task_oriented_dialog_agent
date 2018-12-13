#!/usr/bin/env python3
from utils.internal.arguments import solicit_args
from utils.external.glad_utils import load_dataset, get_models, load_model
import os
import logging
import numpy as np
import torch
from random import seed
import pdb, sys

def run(args):
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed(args.seed)

    dataset, ontology, vocab, Eword = load_dataset()

    model = load_model(args.model, args, ontology, vocab, Eword)
    model.save_config()

    model = model.to(model.device)
    if not args.test_mode:
        logging.info('Starting train')
        model.run_train(dataset['train'], dataset['dev'], args)
    if args.use_existing:
        resume_path = os.path.join(args.report_path, args.model_type, args.prefix)
        model.load_best_save(directory=resume_path)
    else:
        model.load_best_save(directory=args.dout)
    model = model.to(model.device)
    logging.info('Running dev evaluation')
    dev_out = model.run_eval(dataset['dev'], args)
    pprint(dev_out)

if __name__ == '__main__':
    args = solicit_args()
    args.dout = os.path.join(args.report_path, args.model_type, args.suffix)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    args.dropout = {key: args.drop_prob for key in ["emb", "local", "global"]}
    run(args)
