#!/usr/bin/env python3
import os
import logging
import torch

from model.components import device
from model.preprocess import DataLoader
from model.learn import Builder

from utils.internal.arguments import solicit_args

if __name__ == '__main__':
    # args = solicit_args()
    # args.dout = os.path.join(args.report_path, args.model, args.suffix)
    # if not os.path.isdir(args.dout):
        # os.makedirs(args.dout)
    # args.dropout = {key: args.drop_prob for key in ["emb", "local", "global"]}

    # logging.basicConfig(level=logging.INFO)
    # logging.info(args)

    # loader = DataLoader(args)
    # builder = Builder(args, loader, loader.embeddings)
    # glad_model = builder.create_model(len(loader.vocab))

    # if not args.test_mode:
        # logging.info('Starting train')
        # glad_model.run_train(loader.dataset['train'], loader.dataset['dev'], args)
    # if args.use_existing:
        # resume_path = os.path.join(args.report_path, args.model, args.prefix)
        # glad_model.load_best_save(directory=resume_path)
    # else:
        # glad_model.load_best_save(directory=args.dout)
    glad_model = glad_model.to(device)
    logging.info('Running test evaluation')
    test_out = glad_model.run_report(loader.dataset['test'], args)
    print(test_out)
