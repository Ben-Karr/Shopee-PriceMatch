import pandas as pd
import six
from google.cloud import translate_v2 as translate

df = pd.read_csv('src/small_text_df.csv')

text_list = df.title.tolist()
single_txt = text_list[45]
tiny_text_list = text_list[:5]
tiny_df = df.loc[:5].copy()

translate_client = translate.Client()

def translate_text(x):
    #if not isinstance(x, six.binary_type):
    #    return 'Not the right language code'
    
    res = translate_client.translate(x, target_language = 'en')
    
    if res['detectedSourceLanguage'] == 'en':
        return x

    return res['translatedText']

tiny_df['en'] = tiny_df.title.apply(lambda x: translate_text(x))

tiny_df.to_csv('res_tiny_df.csv')