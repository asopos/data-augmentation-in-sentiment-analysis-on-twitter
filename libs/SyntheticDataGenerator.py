from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pandas as pd
import random

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


def generate_synthetic_data(data, synthetic_method, coverage_percentage=0.5, back_translate=None,
                            word_embedding_model=None, tgt_languages=None, gpt2=None, seed_percentage=0.5):
    if synthetic_method == "synonyms":
        return replace_words_with_synonyms(data["Tweet_Token"].values[0], coverage_percentage)
    elif synthetic_method == "back_translation":
        tgt_lang = random.choice(tgt_languages)
        return back_translate.translate_tweet(data["Tweet"].values[0], tgt_lang)
    elif synthetic_method == "word_embedding":
        return replace_words_with_word_embeddings(data["Tweet_Token"].values[0], word_embedding_model,
                                                  coverage_percentage)
    elif synthetic_method == "gp2":
        return gpt2.generate_synthetic_data_with_gpt(data["Tweet_Token"].values[0], seed_percentage)

    return data["Tweet"].values[0]


def fill_synthetic_data_percentage(data, percentage, method, multiplier=1, coverage_percentage=0.5):
    sample_length = int(len(data) * percentage)
    # synth_method = method_mapping[method]
    sample = data.sample(sample_length)
    sample = pd.DataFrame(np.repeat(sample.values, multiplier, axis=0), columns=sample.columns)
    sample["Tweet_Token"] = sample["Tweet"].apply(token_pipeline)
    # sample["Tweet"] = sample["Tweet_Token"].apply((lambda tweet: synth_method(tweet, synonym_percentage)))
    return pd.concat([data, sample])


def fill_missing_labels(data,
                        method,
                        coverage_percentage=0.5,
                        tgt_languages=None,
                        word_embedding_model=None,
                        seed_percentage=0.5):
    data["Tweet_Token"] = data["Tweet"].apply(token_pipeline)
    label_counts = data['Label'].value_counts()
    max_label_count = max(label_counts)
    print(max(label_counts))
    training_dataframes = [data]

    back_translation = BackTranslation(tgt_languages) if method == "back_translation" else None
    gp2 = GPT2() if method == "gp2" else None
    for label in label_counts.index:
        fill_count = max_label_count - label_counts[label]
        label_data = data[data['Label'] == label]
        if fill_count > 0:
            synth_label_data = []
            for i in range(fill_count):
                sample = label_data.sample()
                synth_tweet = generate_synthetic_data(sample, method, coverage_percentage, back_translation,
                                                      word_embedding_model, tgt_languages, gp2, seed_percentage)

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


def replace_words_with_word_embeddings(tokens, model, percentage=0.2):
    tmp_tokens = tokens.copy()
    num_words_to_replace = int(len(tmp_tokens) * percentage)
    words_to_replace = random.sample(range(len(tmp_tokens)), num_words_to_replace)

    for idx in words_to_replace:
        word = tmp_tokens[idx]
        try:
            similar_words = model.most_similar(word, topn=3)
            similar_words = [w for w, _ in similar_words if w.lower() != word.lower()]

            if similar_words:
                new_word = np.random.choice(similar_words)
                tmp_tokens[idx] = new_word
        except KeyError:
            continue
    return " ".join(tmp_tokens)


class BackTranslation:
    def __init__(self, tgt_languages, tgt_lang_combinations=None):
        self.forward_models, self.forward_tokenizers = {}, {}
        self.backward_models, self.backward_tokenizers = {}, {}
        self.tgt_languages = tgt_languages
        self.tgt_lang_combinations = tgt_lang_combinations
        self.translation_cache = {}
        for tgt_lang in tgt_languages:
            forward_model_name = f'Helsinki-NLP/opus-mt-en-{tgt_lang}'
            backward_model_name = f'Helsinki-NLP/opus-mt-{tgt_lang}-en'
            self.forward_tokenizers[("en", tgt_lang)] = MarianTokenizer.from_pretrained(forward_model_name)
            self.forward_models[("en", tgt_lang)] = MarianMTModel.from_pretrained(forward_model_name)
            self.backward_tokenizers[(tgt_lang, "en")] = MarianTokenizer.from_pretrained(backward_model_name)
            self.backward_models[(tgt_lang, "en")] = MarianMTModel.from_pretrained(backward_model_name)

    def back_translate(self, text, forw_tokenizer, forw_model, backw_tokenizer, backw_model):
        forward_input = forw_tokenizer.encode(text, return_tensors="pt")
        forward_output = forw_model.generate(forward_input)
        forward_translation = forw_tokenizer.decode(forward_output[0], skip_special_tokens=True)

        backward_input = backw_tokenizer.encode(forward_translation, return_tensors="pt")
        backward_output = backw_model.generate(backward_input)
        backward_translation = backw_tokenizer.decode(backward_output[0], skip_special_tokens=True)

        return backward_translation

    def multiple_back_translate(self, text, first_forw_tokenizer, first_forw_model, second_forw_tokenizer,
                                second_forw_model, second_backw_tokenizer, second_backw_model, first_backw_tokenizer,
                                first_backw_model):
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

    def translate_tweet(self, tweet, tgt_lang, second_tgt_lang=None):
        cache_key = (tweet, tgt_lang, second_tgt_lang) if second_tgt_lang else (tweet, tgt_lang)

        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        else:
            if second_tgt_lang:
                translation = self.multiple_back_translate(
                    tweet,
                    self.forward_tokenizers[("en", tgt_lang)],
                    self.forward_models[("en", tgt_lang)],
                    self.forward_tokenizers[(tgt_lang, second_tgt_lang)],
                    self.forward_models[(tgt_lang, second_tgt_lang)],
                    self.backward_tokenizers[(second_tgt_lang, tgt_lang)],
                    self.backward_models[(second_tgt_lang, tgt_lang)],
                    self.backward_tokenizers[(tgt_lang, "en")],
                    self.backward_models[(tgt_lang, "en")],
                )
            else:
                translation = self.back_translate(
                    tweet,
                    self.forward_tokenizers[("en", tgt_lang)],
                    self.forward_models[("en", tgt_lang)],
                    self.backward_tokenizers[(tgt_lang, "en")],
                    self.backward_models[(tgt_lang, "en")],
                )
            self.translation_cache[cache_key] = translation
            return translation

    # for tgt_lang_1, tgt_lang_2 in language_combinations:
    #     forward_model_name = f'Helsinki-NLP/opus-mt-{tgt_lang_1}-{tgt_lang_2}'
    #     backward_model_name = f'Helsinki-NLP/opus-mt-{tgt_lang_2}-{tgt_lang_1}'
    #     forward_tokenizers[(tgt_lang_1, tgt_lang_2)] = MarianTokenizer.from_pretrained(forward_model_name)
    #     forward_models[(tgt_lang_1, tgt_lang_2)] = MarianMTModel.from_pretrained(forward_model_name)
    #     backward_tokenizers[(tgt_lang_2, tgt_lang_1)] = MarianTokenizer.from_pretrained(backward_model_name)
    #     backward_models[(tgt_lang_2, tgt_lang_1)] = MarianMTModel.from_pretrained(backward_model_name)


class GPT2:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_synthetic_data_with_gpt(self, tweet_token, seed_percent=0.2, ):
        tokenizer = self.tokenizer
        model = self.model
        length_of_seed_tokens = int(len(tweet_token) * seed_percent)
        seed = " ".join(tweet_token[0:length_of_seed_tokens])
        input_text = tokenizer.encode(seed, return_tensors="pt", padding=True)

        output = model.generate(input_text, max_length=len(tweet_token) * 3, num_return_sequences=1, do_sample=True,
                                temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text.replace("\n", "")
        return generated_text
