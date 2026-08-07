# Given the NER result, we will reconstruct the sentence by converting the labels into complete words

# NER result
ner_result = [
    ('秦', 'B-LOC'), ('始', 'I-LOC'), ('陕', 'B-LOC'), ('西', 'I-LOC'),
    ('省', 'I-LOC'), ('西', 'B-LOC'), ('安', 'I-LOC'), ('市', 'I-LOC'),
    ('国', 'B-ORG'), ('务', 'I-ORG'), ('院', 'I-ORG')
]

# Function to reconstruct words from NER results
def reconstruct_ner_words(ner_result):
    words = []
    current_word = ""
    current_label = ""
    
    for char, label in ner_result:
        if label.startswith("B-"):
            if current_word:  # add the previous word
                words.append((current_word, current_label))
            current_word = char
            current_label = label[2:]
        elif label.startswith("I-") and label[2:] == current_label:
            current_word += char
        else:
            if current_word:  # add the previous word if no continuation
                words.append((current_word, current_label))
            current_word = char
            current_label = label[2:]
    
    if current_word:
        words.append((current_word, current_label))
    
    return words

if __name__ == "__main__":
    # Reconstruct the words
    reconstructed_words = reconstruct_ner_words(ner_result)
    print(reconstructed_words)
