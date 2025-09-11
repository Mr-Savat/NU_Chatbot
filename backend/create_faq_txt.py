# create_faq_txt.py
import csv

input_csv = "./data/faqs.csv"
output_txt = "train.txt"

with open(input_csv, 'r', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as txtfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row if you have one
    for row in reader:
        # Assuming your CSV has two columns: [question, answer]
        if len(row) >= 2:
            question = row[0].strip()
            answer = row[1].strip()
            formatted_line = f"Question: {question} Answer: {answer}<|endoftext|>\n"
            txtfile.write(formatted_line)

print(f"Formatted training data written to {output_txt}")