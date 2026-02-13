from stackapi import StackAPI
import json
from bs4 import BeautifulSoup
import time

# 1. Initialize Gardening StackExchange
# Note: Use 'hydroponic' (singular) for the Gardening site
SITE = StackAPI('gardening')
SITE.page_size = 100
SITE.max_pages = 3 

def clean_html(raw_html):
    if not raw_html: return ""
    return BeautifulSoup(raw_html, "html.parser").get_text()

print("Fetching questions from Gardening StackExchange...")
questions = SITE.fetch('questions', tagged='hydroponic', filter='withbody', sort='votes')

dataset = []

if 'items' in questions:
    for q in questions['items']:
        answer_id = q.get('accepted_answer_id')
        
        # If no accepted answer, try to find the top-voted answer instead
        if not answer_id and q.get('answer_count', 0) > 0:
            print(f"No accepted answer for '{q['title'][:30]}...', fetching top-voted instead.")
            # Fetch answers for this specific question ID
            answers = SITE.fetch(f"questions/{q['question_id']}/answers", filter='withbody', sort='votes')
            if answers.get('items'):
                answer_id = answers['items'][0]['answer_id']

        if answer_id:
            try:
                # Fetch the full text of the chosen answer
                answer_data = SITE.fetch(f'answers/{answer_id}', filter='withbody')
                if answer_data.get('items'):
                    instruction = clean_html(f"{q['title']} {q['body']}")
                    response = clean_html(answer_data['items'][0]['body'])
                    
                    dataset.append({
                        "instruction": instruction,
                        "response": response,
                        "source": q['link']
                    })
                time.sleep(0.2) # Avoid rate limits
            except Exception as e:
                print(f"Error fetching answer {answer_id}: {e}")
                continue

# 3. Save the results
with open('hydro_qa_data.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"\nSuccess! Gathered {len(dataset)} instruction-response pairs.")
