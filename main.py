from argparse import ArgumentParser
from os.path import exists, abspath, isfile, isdir
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder

def get_input_type(generic_input):
    if exists(generic_input) and isfile(generic_input):
        return 'file'
    elif exists(generic_input) and isdir(generic_input):
        return 'folder'
    else:
        return 'url'
    
def prepare_model(model_name, cache_dir='cache', device='auto'):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map=device, config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left')
    return model, tokenizer

def generate(text, tokenizer, model, device='auto'):
    model_in_text_encoded = tokenizer([text], return_tensors='pt').to(device) # padding='max_length', max_length=1024?
    generated_text = model.generate(**model_in_text_encoded) # max_length=1024, num_return_sequences=1)
    return tokenizer.decode(generated_text, skip_special_tokens=True)[0]

def main(args):
    if args.token is not None:
        HfFolder.save_token(args.token)

    input_type = get_input_type(args.input)
    model, tokenizer = prepare_model(args.model, args.cache, args.device)
    # TODO prepare
    print('predict')
    print('ciao sono un testo', generate('ciao sono un testo', tokenizer, model, args.device))

if __name__ == '__main__':
    aparse = ArgumentParser()
    aparse.add_argument('--cache', type=str, default='cache')
    aparse.add_argument('-i', '--input', type=str, required=True)
    aparse.add_argument('-m', '--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
    aparse.add_argument('-d', '--device', type=str, default='auto')
    aparse.add_argument('--token', type=str, default=None)

    main(aparse.parse_args())