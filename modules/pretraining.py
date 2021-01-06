from LAMA.lama.modules import build_model_by_name
import torch 
import pandas as pd

import modules.options_bert
language_model = 'bert'


def bert_sentence_contextual_embedding(sentences, *args):
    
    # print('Getting BERT contextual embedding')
    parser = modules.options_bert.get_general_parser()
    args = modules.options_bert.parse_args(parser)
    
    model = build_model_by_name(language_model, args)
    contextual_embeddings, sentence_lengths, tokenized_text_list = model.get_contextual_embeddings([[sentences]])
    
    # contextual_embeddings has shape : [ # layers, # batches, # tokens, # features], e.g. [12 x 1 x 22 x 768]

    # since contextual_embeddings is a list of tensors, I first stack them together:
    token_embeddings = torch.stack(contextual_embeddings, dim=0)
    # token_embeddings.size() = torch.Size([12, 1, 16, 768])
    
    # remove the batch dimension, since its useless in this case:
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # token_embeddings.size() = torch.Size([12, 16, 768])

    # Average the second to last hidden layer of each token producing a single 768 length vector
    sentence_embedding = list(torch.mean(token_embeddings, dim=(0, 1)))

    sentence_embedding = [x for x in sentence_embedding]
    return sentence_embedding


def bert_contextual_embedding(df):
    """
    Convert the text of the tweets to its BERT contextual representation
    Parameters
    ----------
    df : Dataframe
        Dataframe with the tweets to encode.

    Returns
    -------
    tweet_embedding : Dataframe
        Dataframe with the tweets encoded in the 768 bert features
    """
    tweet_embedding = df.apply(bert_sentence_contextual_embedding)
    tweet_embedding = pd.DataFrame(tweet_embedding, columns=['tweet'], index=tweet_embedding.index)
    tweet_embedding = pd.DataFrame(tweet_embedding.tweet.tolist(), columns=list(range(1, 769)), index=tweet_embedding.index)
    return tweet_embedding
