import jsonlines
import logging
from tqdm import tqdm

from transformers import(
	MT5ForConditionalGeneration,
	T5Tokenizer,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
)

import torch
from torch.utils.data import DataLoader

from dataset import NewsDataset
from argparse import ArgumentParser, Namespace
from pathlib import Path

logging.basicConfig(
	format="%(asctime)s | %(levelname)s | %(message)s",
	level=logging.INFO,
	datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
	#load data
	with jsonlines.open(args.input_file) as f:
		loading_file = []
		for line in f.iter():
			loading_file.append(line)
		data = loading_file

	logging.info("data loaded")

	#clean text
	for i in range(len(data)):
		data[i]['title'] = data[i]['title'].replace('\n', '')
		data[i]['maintext'] = data[i]['maintext'].replace('\n', '')

	logging.info('text cleaned')

	#create model and tokenizer
	model = MT5ForConditionalGeneration.from_pretrained("./checkpoint-10856")
	tokenizer = T5Tokenizer.from_pretrained("./checkpoint-10856", use_fast=True)

	logging.info('model loaded')

	#creat dataset
	eval_dataset = NewsDataset(data, tokenizer)

	#train
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print('device: ', device)

	model.to(device)
	model.eval()

	news_title = []
	eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=eval_dataset.collate_fn)

	for num, X in enumerate(tqdm(eval_loader)):
		encoder_inputs_ids = X['input_ids']
		attention_mask = X['attention_mask']
		outputs = model.generate(encoder_inputs_ids.to(device), num_beams=2, max_length=30, attention_mask=attention_mask.to(device))
		outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
		for i in range(len(outputs)):
			news_title.append(outputs[i])


	#create ouput data
	outputs_data = []
	for i in range(len(news_title)):
		news = {}
		news['title'] = news_title[i]
		news['id'] = data[i]['id']
		outputs_data.append(news)

	with jsonlines.open(args.output_file, mode='w') as writer:
		writer.write_all(outputs_data)


def parse_args() -> Namespace:
	parser = ArgumentParser()

	parser.add_argument(
		"--input_file",
		type=Path,
		help='path to input file',
		default="./data/public.jsonl"
	)

	parser.add_argument(
		"--output_file",
		type=Path,
		help='path to output file',
		default="./public_predict.jsonl"
	)

	parser.add_argument(
		"--batch_size",
		type=int,
		help='training batch size',
		default=2,
	)

	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = parse_args()
	#args.output_dir.mkdir(parents=True, exist_ok=True)
	main(args)

