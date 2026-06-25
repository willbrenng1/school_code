import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")


class Preprocess(object):
    def __init__(self):
        self.regex = (
            "^\s+|\W+|[0-9]|\s+$"  # for autograder consistency, please do not change
        )

    def clean_text(self, text: str) -> str:
        """
        Clean the input text string:
            1. Remove HTML formatting (use Beautiful Soup)
            2. Remove non-alphabet characters such as punctuation or numbers and replace with ' '
               You may refer back to the slides for this part (For autograder consistency,
               we implement this part for you, please do not change it.)
            3. Remove leading or trailing white spaces including any newline characters
            4. Convert to lower case
            5. Tokenize and remove stopwords using nltk's 'english' vocabulary
            6. Rejoin remaining text into one string using " " as the word separator

        Args:
            text: string

        Return:
            cleaned_text: string
        """
        # 1. Remove HTML formatting (use Beautiful Soup)
        cleaned_text = BeautifulSoup(text, "html.parser").get_text()

        # Step 2 is implemented for you, please do not change
        cleaned_text = re.sub(self.regex, " ", cleaned_text).strip()
        # 3. Remove leading or trailing white spaces including any newline characters
        cleaned_text = cleaned_text.strip()
        # 4. Convert to lower case
        cleaned_text = cleaned_text.lower()
        # 5. Tokenize and remove stopwords using nltk's 'english' vocabulary
        # 6. Rejoin remaining text into one string using " " as the word separator
        stop_words = set(stopwords.words("english"))
        str_tok = word_tokenize(cleaned_text)

        return " ".join([w for w in str_tok if w not in stop_words])

    def clean_dataset(self, data: list[str]) -> list[str]:
        """
        Given an array of strings, clean each string in the array by calling clean_text()

        Args:
            data: list of N strings

        Return:
            cleaned_data: list of cleaned N strings
        """
        return [self.clean_text(e) for e in data]


# Note that clean_wos is outside of the Preprocess class
def clean_wos(x_train: list[str], x_test: list[str]) -> tuple[list[str], list[str]]:
    """
    ToDo: Clean both the x_train and x_test dataset using clean_dataset from Preprocess

    Input:
        x_train: list of N strings
        x_test: list of M strings

    Output:
        cleaned_text_wos: list of cleaned N strings
        cleaned_text_wos_test: list of cleaned M strings
    """
    cleaned_text_wos_class = Preprocess()
    cleaned_text_wos = cleaned_text_wos_class.clean_dataset(x_train)

    cleaned_text_wos_test_class = Preprocess()
    cleaned_text_wos_test = cleaned_text_wos_test_class.clean_dataset(x_test)

    return tuple([cleaned_text_wos, cleaned_text_wos_test])
