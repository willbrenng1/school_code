class Preprocess_Test():
    def __init__(self):

        # Sample data: 
        self.x_train = ["A 65-Year-Old Man's Typewriter Was <strong>Destroyed</strong> By An Angry Cop, And The Internet Got Him A New One And The Internet was helpful &nbsp;\n",
                         'Can You Identify These 5 UNITED STATES Leaders????\n',
                         'Index of Economic Activity Declined in March\n',
                         "2015's Best News Bloopers Are Here And They're Out Of Control #@$%^$ ##OMG\n",
                         '18 Pictures Everyone Who Loves Spilling The Tea Will Understand\n',
                         'This Photographer Celebrates The Diversity Of Black Hair With An Adorable Series of Black and White Photos\n']


        # Sample outputs for misc helper functions:
        self.cleaned_text = ['year old man typewriter destroyed angry cop internet got new one internet helpful',
                             'identify united states leaders',
                             'index economic activity declined march',
                             'best news bloopers control omg',
                             'pictures everyone loves spilling tea understand',
                             'photographer celebrates diversity black hair adorable series black white photos']
        
