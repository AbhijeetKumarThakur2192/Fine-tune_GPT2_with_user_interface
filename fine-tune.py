# PLEASE DON'T CHANGE THE CODE UNLESS YOU'RE FAMILIAR WITH PROGRAMMING. IT MIGHT CAUSE ERRORS OTHERWISE.
# IMPORT NECESSARY MODULES 
import re
import os
import torch
import transformers
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling

# DEFINE FUNCTIONS
def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def load_dataset(file_path, tokenizer, block_size):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def get_model_name():
    return input("Enter the model name to use (e.g., 'gpt2'): ")

def get_batch_size():
    return int(input("Enter the batch size: "))

def get_num_epochs():
    return float(input("Enter the number of epochs: "))

def get_output_dir():
    return input("Enter the output directory path: ")

def get_data_file_path():
    return input("Enter the path of your data.txt file: ")

def get_save_steps():
    return int(input("Enter the number of steps: "))

def get_block_size():
    return int(input("Enter the block size: "))

def get_learning_rate():
    return float(input("Enter the learning rate: "))

def train(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, learning_rate, block_size):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# PROCESSING CODE
    text_data = read_txt(train_file_path)
    text_data = re.sub(r'\n+', '\n', text_data).strip()

    train_dataset = load_dataset(train_file_path, tokenizer, block_size)

    tokenizer.save_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        learning_rate=learning_rate,
        do_train=True  
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), 
            train_dataset=train_dataset
        )

        trainer.train()
        trainer.save_model()
        print("Your fine-tuning process is completed. Now you can use your fine-tuned model.")        
    except Exception as e:
        print(f"Oh! It seems like something went wrong: {str(e)}. Please check all information again or open GitHub issue!")

# MAIN FUNCTION FOR USER INPUT
def main():
    
    while True:
    	print("REMEMBER BEFORE FINE-TUNE YOU MUST HAVE NECESSARY MODULES LIKE PYTORCH, TRANSFORMERS, ACCELERATE IF NOT PLEASE INSTALL THEM OTHERWISE YOU GET ERROR")
    
        print("Type 1 to start fine-tune process or type 2 for EXIT")
        print("\nMenu:")
        print("1. Fine-tune the model")
        print("2. Exit")

        choice = input("Enter your choice (1 or 2): ")
        if choice == '1':         
            model_name = get_model_name()
            data_txt = get_data_file_path()
            output_dir = get_output_dir()
            batch_size = get_batch_size()
            num_epochs = get_num_epochs()
            block_size = get_block_size()
            learning_rate = get_learning_rate()
            save_steps = get_save_steps()
            

            train(
                train_file_path=data_txt,
                model_name=model_name,
                output_dir=output_dir,
                overwrite_output_dir=False,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                save_steps=save_steps,
                learning_rate=learning_rate,
                block_size=block_size
            )
            break
            
        elif choice == '2':
            print("Bye")
            break
            
        else:
            print("Please type 1 or 2")

if __name__ == "__main__":
    main()