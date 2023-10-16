import re

def split_punctuation(text):
    # Split the text into words and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)

    # Initialize variables
    new_text = ""
    mapping = []

    for i, token in enumerate(tokens):
        # If the token is a word, add it to the new text and update the mapping
        if token.isalnum():
            new_text += token + " "
            mapping.extend([i] * len(token))
        # If the token is punctuation, add it to the new text
        else:
            new_text += token + " "

    # Remove the trailing space
    new_text = new_text.rstrip()

    return new_text, mapping

# Input text
input_text = "Hôm nay trời đẹp nhỉ! Mình cùng đi chơi nhé."

# Split punctuation and get the mapping
new_text, index_mapping = split_punctuation(input_text)

# Print the modified text and the index mapping
print(new_text)
print("Index mapping:", index_mapping)
