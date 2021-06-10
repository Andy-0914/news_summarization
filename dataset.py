import torch

class NewsDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer):
		self.data = data
		self.tokenizer = tokenizer

		enoceder_text = [sample['maintext'] for sample in self.data]
		encoder_title = [sample['title'] for sample in self.data]


	def __getitem__(self, idx):
		return self.data[idx]


	def __len__(self):
		return len(self.data)


	def collate_fn(self, samples):
		enoceder_text = [sample['maintext'] for sample in samples]
		encoder_title = [sample['title'] for sample in samples]

		encoder_inputs = self.tokenizer(enoceder_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
		with self.tokenizer.as_target_tokenizer():
			labels = self.tokenizer(encoder_title, return_tensors="pt", padding=True, truncation=True, max_length=256)

		inputs = {'input_ids': encoder_inputs['input_ids'], 
					'attention_mask': encoder_inputs['attention_mask'],
					'labels': labels}
		return inputs


