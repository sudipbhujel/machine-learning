# Importing Libraries
import re
import pandas as pd
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

# Defining Corpus
corpus = [("American Airlines", "ORGANIZATION"), ("Bengaluru", "LOCATION"),
          ("Amazon", "ORGANIZATION"), ("Microsoft", "ORGANIZATION"),
          ("COVID-19", "VIRUS"), ("Florida", "LOCATION"), ("COVID", "VIRUS"),
          ("Covid", "VIRUS"), ("Marcus Deion Brown", "PERSON"),
          ("Trump", "CANDIDATE"), ("Joe Biden", "CANDIDATE"),
          ("America", "COUNTRY"), ("Black American", "PEOPLE"),
          ("Stephen Miller", "PERSON"), ("Orlando", "LOCATION"),
          ("Robin", "PERSON"), ("SouthDaytona", "LOCATION"),
          ("GeorgeFloyd", "PERSON"), ("WHO", "ORGANIZATION"),
          ("Georgia", "LOCATION"), ("Rona", "Person"),
          ("US", "COUNTRY"), ("America", "COUNTRY"),
          ("Seattle", "LOCATION"), ("Tulsa", "LOCATION")]


def cleaned_text(text):
    # Find URL
    url_pattern = re.compile(r'https?:\S+|www\.\S+')

    # Find emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE
                               )

    # Find Emoticons
    emoticon_pattern = re.compile(
        u'(' + u'|'.join(k for k in EMOTICONS) + u')')

    # Twitter processing
    handle_pattern = re.compile(r'@([A-Za-z0-9_]+)')
    rt_pattern = re.compile(r'RT :')
    tag_pattern = re.compile(r'#([A-Za-z0-9_]+)')

    # List of regex
    filters = [url_pattern, emoji_pattern,
               emoticon_pattern, handle_pattern, rt_pattern, tag_pattern]

    string = text

    for filter in filters:
        string = filter.sub(r'', string)
    return string


def entity_extraction(text):
    entities = map(lambda x: x[0], corpus)
    prepare_list = []
    entities = list(map(lambda x: x[0], corpus))
    for entity in entities:
        if entity in text:
            prepare_list.append(corpus[entities.index(entity)])
    if len(prepare_list) == 0:
        return None
    return prepare_list


if __name__ == "__main__":
    # import dataset
    print("Loading Dataset...")
    df = pd.read_csv("EnglishTweets.csv")
    df.drop(df.columns[df.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)

    print("Processing...")
    # cleaned_text
    df["cleaned_text"] = df["text"].apply(cleaned_text)

    # extract entities
    df["extracted_entities"] = df["cleaned_text"].apply(entity_extraction)

    # Save result to csv file
    df.to_csv('op.csv')

    # Print Status
    print("Done! Check op.csv File")
