import numpy as np

class BagofWords_Test():
    def __init__(self):

        # Sample data: 
        self.cleaned_text = ['year old man typewriter destroyed angry cop internet got new one',
                             'identify united states leaders',
                             'index economic activity declined march',
                             'best news bloopers control omg',
                             'pictures everyone loves spilling tea understand',
                             'photographer celebrates diversity black hair adorable series black white photos']
        

        # Sample outputs for misc helper functions:
        self.data_split = [['year', 'old', 'man', 'typewriter', 'destroyed', 'angry', 'cop',
                        'internet', 'got', 'new', 'one'], ['identify', 'united', 'states', 'leaders'], 
                        ['index', 'economic', 'activity', 'declined', 'march'], ['best', 'news', 'bloopers', 
                        'control', 'omg'], ['pictures', 'everyone', 'loves', 'spilling', 'tea', 'understand'], 
                        ['photographer', 'celebrates', 'diversity', 'black', 'hair', 'adorable', 'series', 'black', 'white', 'photos']]

        self.flattened_list = ['year', 'old', 'man', 'typewriter', 'destroyed', 'angry', 'cop',
                            'internet', 'got', 'new', 'one', 'identify', 'united', 'states',
                            'leaders', 'index', 'economic', 'activity', 'declined', 'march',
                            'best', 'news', 'bloopers', 'control', 'omg', 'pictures',
                           'everyone', 'loves', 'spilling', 'tea', 'understand', 'photographer', 
                           'celebrates', 'diversity', 'black', 'hair', 'adorable', 'series', 'black', 
                           'white', 'photos']
        self.fitted_categories = ['activity', 'adorable', 'angry', 'best', 'black', 'bloopers', 'celebrates',
                            'control', 'cop', 'declined', 'destroyed', 'diversity', 'economic', 'everyone',
                            'got', 'hair', 'identify', 'index', 'internet', 'leaders', 'loves', 'man',
                            'march', 'new', 'news', 'old', 'omg', 'one', 'photographer', 'photos', 'pictures',
                            'series', 'spilling', 'states', 'tea', 'typewriter', 'understand', 'united',
                            'white', 'year']

        self.vocab_size = 40

        self.encode_0 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        self.cleaned_text_bow = np.array([[0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
                                0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0.],
                                [0., 1., 0., 0., 2., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.]])




