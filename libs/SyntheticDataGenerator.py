from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import pandas as pd
import random
import re

tweet_tokenizer = TweetTokenizer()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def token_pipeline(tweet):
    # Lowercase the tweet
    tweet = tweet.lower()

    # Remove URLs
    #    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)

    # Remove user mentions
    #    tweet = re.sub(r'\@\w+', '', tweet)

    # Remove hashtags
    #    tweet = re.sub(r'\#\w+', '', tweet)

    # Remove special characters and punctuation
    #    tweet = re.sub(r'\W', ' ', tweet)

    # Remove digits and numbers
    #    tweet = re.sub(r'\d', '', tweet)

    tokens = tweet_tokenizer.tokenize(tweet)

    #    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens


def fill_synthetic_data_percentage(data, percentage, method, multiplier=1, synonym_percentage=0.5):

    sample_length = int(len(data) * percentage)
    synth_method = method_mapping[method]
    sample = data.sample(sample_length)
    sample = pd.DataFrame(np.repeat(sample.values, multiplier, axis=0), columns=sample.columns)
    sample["Tweet_Token"] = sample["Tweet"].apply(token_pipeline)
    sample["Tweet"] = sample["Tweet_Token"].apply((lambda tweet: synth_method(tweet, synonym_percentage)))
    return pd.concat([data, sample])

def fill_missing_labels(data, method, synonym_percentage=0.5):
    data["Tweet_Token"] = data["Tweet"].apply(token_pipeline)
    label_counts = data['Label'].value_counts()
    max_label_count = max(label_counts)
    synth_method = method_mapping[method]
    print(max(label_counts))
    training_dataframes = [data]

    for label in label_counts.index:
        fill_count = max_label_count - label_counts[label]
        label_data = data[data['Label'] == label]
        if fill_count > 0:
            synth_label_data = []
            for i in range(fill_count):
                sample = label_data.sample()
                synth_tweet = synth_method(sample["Tweet_Token"].values[0], synonym_percentage)
                label = sample["Label"].values[0]
                synth_label_data.append([synth_tweet, label])
            synth_label_data = pd.DataFrame(synth_label_data, columns=["Tweet", "Label"])
            training_dataframes.append(synth_label_data)
        print(label, label_counts[label])
    return pd.concat(training_dataframes)



def pos_mapping(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return None


def replace_words_with_synonyms(tweet_tokens, percentage=0.2, similarity_threshold=0.8):
    tmp_tokens = tweet_tokens.copy()
    tokens_pos = [pos[1] for pos in pos_tag(tmp_tokens)]
    num_to_replace = int(len(tmp_tokens) * percentage)

    for _ in range(num_to_replace):
        rand_index = random.randint(0, len(tmp_tokens) - 1)

        word = tmp_tokens[rand_index]
        pos = tokens_pos[rand_index]
        synsets = wordnet.synsets(word, pos_mapping(pos))
        original_synset = synsets[0] if synsets else None
        similar_synonyms = set()

        for synset in synsets:
            for lemma in synset.lemmas():
                lemma_synset = lemma.synset()
                similarity = original_synset.wup_similarity(lemma_synset)
                if similarity and similarity >= similarity_threshold:
                    similar_synonyms.add(lemma.name())

        if len(similar_synonyms) > 0:
            tmp_tokens[rand_index] = random.choice(list(similar_synonyms))

    return ' '.join(tmp_tokens)

tgt_languages = [
    "fr",
    "de",
#    "es",
    "ru",
#    "jap"
]
language_combinations = [
#    ("fr", "de"),
#    ("fr", "ru"),
#    ("de", "fr"),
#    ("de", "ru"),
#    ("ru", "de")
]

def back_translate(text, forw_tokenizer, forw_model, backw_tokenizer, backw_model):
    forward_input = forw_tokenizer.encode(text, return_tensors="pt")
    forward_output = forw_model.generate(forward_input)
    forward_translation = forw_tokenizer.decode(forward_output[0], skip_special_tokens=True)

    backward_input = backw_tokenizer.encode(forward_translation, return_tensors="pt")
    backward_output = backw_model.generate(backward_input)
    backward_translation = backw_tokenizer.decode(backward_output[0], skip_special_tokens=True)

    return backward_translation

def multiple_back_translate(text, first_forw_tokenizer, first_forw_model, second_forw_tokenizer, second_forw_model, second_backw_tokenizer, second_backw_model, first_backw_tokenizer, first_backw_model):
    first_forward_input = first_forw_tokenizer.encode(text, return_tensors="pt")
    first_forward_output = first_forw_model.generate(first_forward_input)
    first_forward_translation = first_forw_tokenizer.decode(first_forward_output[0], skip_special_tokens=True)

    second_forward_input = second_forw_tokenizer.encode(first_forward_translation, return_tensors="pt")
    second_forward_output = second_forw_model.generate(second_forward_input)
    second_forward_translation = second_forw_tokenizer.decode(second_forward_output[0], skip_special_tokens=True)

    second_backward_input = second_backw_tokenizer.encode(second_forward_translation, return_tensors="pt")
    second_backward_output = second_backw_model.generate(second_backward_input)
    second_backward_translation = second_backw_tokenizer.decode(second_backward_output[0], skip_special_tokens=True)

    first_backward_input = first_backw_tokenizer.encode(second_backward_translation, return_tensors="pt")
    first_backward_output = first_backw_model.generate(first_backward_input)
    first_backward_translation = first_backw_tokenizer.decode(first_backward_output[0], skip_special_tokens=True)

    return first_backward_translation

forward_models, forward_tokenizers = {}, {}
backward_models, backward_tokenizers = {}, {}

# for tgt_lang in tgt_languages:
#     forward_model_name = f'Helsinki-NLP/opus-mt-en-{tgt_lang}'
#     backward_model_name = f'Helsinki-NLP/opus-mt-{tgt_lang}-en'
#     forward_tokenizers[("en",tgt_lang)] = MarianTokenizer.from_pretrained(forward_model_name)
#     forward_models[("en",tgt_lang)] = MarianMTModel.from_pretrained(forward_model_name)
#     backward_tokenizers[(tgt_lang, "en")] = MarianTokenizer.from_pretrained(backward_model_name)
#     backward_models[(tgt_lang, "en")] = MarianMTModel.from_pretrained(backward_model_name)
#
# for tgt_lang_1, tgt_lang_2 in language_combinations:
#     forward_model_name = f'Helsinki-NLP/opus-mt-{tgt_lang_1}-{tgt_lang_2}'
#     backward_model_name = f'Helsinki-NLP/opus-mt-{tgt_lang_2}-{tgt_lang_1}'
#     forward_tokenizers[(tgt_lang_1, tgt_lang_2)] = MarianTokenizer.from_pretrained(forward_model_name)
#     forward_models[(tgt_lang_1, tgt_lang_2)] = MarianMTModel.from_pretrained(forward_model_name)
#     backward_tokenizers[(tgt_lang_2, tgt_lang_1)] = MarianTokenizer.from_pretrained(backward_model_name)
#     backward_models[(tgt_lang_2, tgt_lang_1)] = MarianMTModel.from_pretrained(backward_model_name)


translation_cache = {}

def translate_tweet(tweet, tgt_lang, second_tgt_lang=None):
    cache_key = (tweet, tgt_lang, second_tgt_lang) if second_tgt_lang else (tweet, tgt_lang)

    if cache_key in translation_cache:
        return translation_cache[cache_key]
    else:
        if second_tgt_lang:
            translation = multiple_back_translate(
                tweet,
                forward_tokenizers[("en", tgt_lang)],
                forward_models[("en", tgt_lang)],
                forward_tokenizers[(tgt_lang, second_tgt_lang)],
                forward_models[(tgt_lang, second_tgt_lang)],
                backward_tokenizers[(second_tgt_lang, tgt_lang)],
                backward_models[(second_tgt_lang, tgt_lang)],
                backward_tokenizers[(tgt_lang, "en")],
                backward_models[(tgt_lang, "en")],
            )
        else:
            translation = back_translate(
                tweet,
                forward_tokenizers[("en", tgt_lang)],
                forward_models[("en", tgt_lang)],
                backward_tokenizers[(tgt_lang, "en")],
                backward_models[(tgt_lang, "en")],
            )
        translation_cache[cache_key] = translation
        return translation

method_mapping={
    "synonyms": replace_words_with_synonyms,
    "back_translation": translate_tweet
}