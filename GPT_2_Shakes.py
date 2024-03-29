import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "gpt_2_news.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/de0015/GPT_Data/main/GPT-2_news.txt"
	data = requests.get(url)

	with open(file_name, 'w') as f:
		f.write(data.text)


sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=10)   # steps is max number of training steps

gpt2.generate(sess)


import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(sess)
single_text = gpt2.generate(sess, return_as_list=True)[0]
print(single_text)
output_news.write(single_text)


