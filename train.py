import jsonlines
import logging

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

class NewsTrainer(Seq2SeqTrainer):
	def compute_loss(self, model, inputs, return_outputs=False):

		labels = inputs['labels']

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		outputs = model(input_ids=inputs['input_ids'].to(device), 
						attention_mask=inputs['attention_mask'].to(device),
						labels=labels['input_ids'].to(device),
						decoder_attention_mask=labels['attention_mask'].to(device))

		return outputs.loss
		

logging.basicConfig(
	format="%(asctime)s | %(levelname)s | %(message)s",
	level=logging.INFO,
	datefmt="%Y-%m-%d %H:%M:%S",
)

TRAIN = "train"
PUBLIC = "public"
SPLITS = [TRAIN, PUBLIC]

def main(args):
	#load data
	data = {}
	for split in SPLITS:
		with jsonlines.open('data/{}.jsonl'.format(split)) as f:
			loading_file = []
			for line in f.iter():
				loading_file.append(line)
			data[split] = loading_file

	logging.info("data loaded")

	#clean text
	for split in SPLITS:
		for i in range(len(data[split])):
			data[split][i]['title'] = data[split][i]['title'].replace('\n', '')
			data[split][i]['maintext'] = data[split][i]['maintext'].replace('\n', '')

	logging.info('text cleaned')

	#create model and tokenizer
	model = MT5ForConditionalGeneration.from_pretrained("./results/checkpoint-16284")
	tokenizer = T5Tokenizer.from_pretrained("./results/checkpoint-16284")

	logging.info('model loaded')

	#creat dataset
	train_dataset = NewsDataset(data['train'], tokenizer)
	eval_dataset = NewsDataset(data['public'], tokenizer)

	#train
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)


	training_args = Seq2SeqTrainingArguments(
		output_dir=args.output_dir,
		num_train_epochs=2,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		warmup_steps=500,
		weight_decay=0.01,
		logging_dir=args.logging_dir,
		logging_steps=10,
		learning_rate=1e-4,
		save_strategy=args.save_strategy,
	)

	trainer = NewsTrainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		data_collator=train_dataset.collate_fn, 
		train_dataset=train_dataset,
	)

	logging.info('start training')

	trainer.train()



def parse_args() -> Namespace:
	parser = ArgumentParser()

	parser.add_argument(
		"--output_dir",
		type=Path,
		help="Directory to output model and tokenizer",
		default="./results",
	)

	parser.add_argument(
		"--logging_dir",
		type=Path,
		help="Directory to logging information",
		default="./logs"
	)

	parser.add_argument(
		"--save_strategy",
		type=str,
		help="saving saving strategy for training",
		default="epoch",
	)

	parser.add_argument(
		"--batch_size",
		type=int,
		help='training batch size',
		default=4,
	)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main(args)

