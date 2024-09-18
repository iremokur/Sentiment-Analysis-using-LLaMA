import ollama
import json
import pandas as pd


# Define a function to process each text entry with Ollama
def process_with_ollama(text, model_prompt):
    # Create the full prompt
    full_prompt = f"{model_prompt}: {text}"

    # Send the prompt to Ollama and get the response
    response = ollama.generate(model="llama3.1", prompt=full_prompt)

    # Extract the relevant response content, assuming Ollama's output is a dict with 'output' key
    output = response.get('response', 'No response received')

    results = []

    results.append({
        "full_prompt": full_prompt,
        "output": output
    })

    return results

if __name__ == '__main__':
    # Define the prompt you want to use with each text element
    df = pd.read_csv("EV_reviews.csv", on_bad_lines='skip', engine='python')

    model_prompt = "Translate the following text to English and then Determine the sentiment of every text and provide a sentiment label for every text as negative,  positive , neutral"


    # Define batch size
    batch_size = 5 # Adjust based on model limits

    # Split DataFrame into batches
    texts = df['Review Text'].tolist()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Process each batch and collect results
    all_results = []
    for batch in batches:
        batch_results = process_with_ollama(batch, model_prompt)
        all_results.extend(batch_results)

    # Convert the results into JSON format
    results_json = json.dumps(all_results, indent=4)
    # Save the JSON to a file
    output_filename = 'results.json'
    with open(output_filename, 'w') as json_file:
        json_file.write(results_json)
