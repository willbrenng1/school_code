def test_init_dataset(dataset_type):
    if ('IterableDataset' in dataset_type):
        print("Dataset type test passed \n")
    else:
        print("Dataset test faied. Type of dataset should be Iterable Dataset. Your type is: ", dataset_type, "\n")

def test_init_tokenizer(tokenizer_type):
    if ('DistilBertTokenizerFast' in tokenizer_type):
        print("Init tokenizer test passed \n")
    else: 
        print("Type of tokenizer should be Dilbert Tokenizer Fast. Your type is: ", tokenizer_type, "\n")

tokenizer_example_sentences_case1 = {"text": ["This sentence is the truth!!", "This sentence isn't the truth.", "One of these sentences isn't a truth?"]}
# Define the new sentence with more than 15 words
tokenizer_example_sentences_case2 = {"text": ["This sentence is the truth!!", "This sentence isn't the truth.", "One of these sentences isn't a truth?","During the early morning hours, the quiet town was full of activity as people prepared for the upcoming festival, filling the streets with vibrant colors and delightful scents."]}

def test_tokenize(tokenizer_output_case1, tokenizer_output_case2):
    
    #older dict
    # example_tokens_dict = {'tokens': [['[CLS]', 'This', 'sentence', 'is', 'the', 'truth', '!', '!', '[SEP]', '[PAD]'], ['[CLS]', 'This', 'sentence', 'isn', "'", 't', 'the', 'truth', '.', '[SEP]'], ['[CLS]', 'One', 'of', 'these', 'sentences', 'isn', "'", 't', 'a', '[SEP]']]}
    
    #updated dict
    example_tokens_dict_case1 = {'tokens': [['[CLS]',
   'This',
   'sentence',
   'is',
   'the',
   'truth',
   '!',
   '!',
   '[SEP]',
   '[PAD]',
   '[PAD]',
   '[PAD]',
   '[PAD]',
   '[PAD]',
   '[PAD]'], 
   ['[CLS]',
  'This',
  'sentence',
  'isn',
  "'",
  't',
  'the',
  'truth',
  '.',
  '[SEP]',
  '[PAD]',
  '[PAD]',
  '[PAD]',
  '[PAD]',
  '[PAD]'], 
   ['[CLS]',
  'One',
  'of',
  'these',
  'sentences',
  'isn',
  "'",
  't',
  'a',
  'truth',
  '?',
  '[SEP]',
  '[PAD]',
  '[PAD]',
  '[PAD]']
  ]}
    
    # Update the expected tokens dictionary to include the new tokens
    global new_sentence_tokens
    example_tokens_dict_case2 = {'tokens': [['[CLS]',
   'This',
   'sentence',
   'is',
   'the',
   'truth',
   '!',
   '!',
   '[SEP]',
   '[PAD]',
   '[PAD]',
   '[PAD]',
   '[PAD]',
   '[PAD]',
   '[PAD]'], 
   ['[CLS]',
  'This',
  'sentence',
  'isn',
  "'",
  't',
  'the',
  'truth',
  '.',
  '[SEP]',
  '[PAD]',
  '[PAD]',
  '[PAD]',
  '[PAD]',
  '[PAD]'], 
   ['[CLS]',
  'One',
  'of',
  'these',
  'sentences',
  'isn',
  "'",
  't',
  'a',
  'truth',
  '?',
  '[SEP]',
  '[PAD]',
  '[PAD]',
  '[PAD]'],['[CLS]',
  'During',
  'the',
  'early',
  'morning',
  'hours',
  ',',
  'the',
  'quiet',
  'town',
  'was',
  'full',
  'of',
  'activity',
  '[SEP]']
  ]}

    print(f'Expected Output for case 1: {example_tokens_dict_case1}\n')
    print(f'Your Output for case 1: {tokenizer_output_case1}\n')
    
    print(f'Expected Output for case 2: {example_tokens_dict_case2}\n')
    print(f'Your Output for case 2: {tokenizer_output_case2}\n')

    max_len = 15

    if ('tokens' not in tokenizer_output_case2.keys()):
        print("Test failed, please check the structure of what you are returning")
        return
    if (len(tokenizer_output_case2['tokens']) != 4):
        print("Test failed, please check to make sure you are handling batching")
        return
    if (len(tokenizer_output_case2['tokens'][0]) != max_len):
        print("Test failed, please check to make sure you are handling padding/truncation properly")
        return
    if (len(tokenizer_output_case2['tokens'][3]) != max_len):  # Updated to check the token length of the new sentence
        # print(len(tokenizer_output['tokens'][3]))
        print("Test failed, please check to make sure you are handling longer sequences properly")
        return
    token_lengths = [len(tokens) for tokens in tokenizer_output_case1['tokens']]
    if not all(length == max_len for length in token_lengths):
        print("Test failed, please make sure all tokens are padded to max_length.")
        return
    print("\nTokenizer functionality tests passed! \n")

def test_preprocess(first_row):
    first_token = first_row['tokens'][0]
    second_token = first_row['tokens'][1]
    if second_token == 'Ana' and first_token == '[CLS]':
        print("First row test passed")
    else:
        print("Preprocess test failed")




