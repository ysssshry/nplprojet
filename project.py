from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

def generate_title_using_t5(abstract):
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Prepare the input text with the prefix "summarize:"
    input_text = "summarize: " + abstract
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the title with better parameters
    outputs = model.generate(
        inputs,
        max_length=15,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        attention_mask=inputs.ne(tokenizer.pad_token_id)
    )

    # Decode the generated title
    generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_title.strip()

def process_file(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return
    
    try:
        # Read the abstract from file
        with open(file_path, 'r', encoding='utf-8') as file:
            abstract = file.read().strip()
        
        # Generate title
        generated_title = generate_title_using_t5(abstract)
        
        # Print results
        print("\nFile Content:")
        print("-" * 50)
        print(abstract[:200] + "..." if len(abstract) > 200 else abstract)
        print("\nGenerated Title:")
        print("-" * 50)
        print(generated_title)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

# Example usage
if __name__ == "__main__":
    # You can change this to any text file you want to process
    file_path = "abstract.txt"  # The file should be in the same directory or provide full path
    
    # Process the file
    process_file(file_path)
