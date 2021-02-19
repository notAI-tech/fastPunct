# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:46:01 2020

@author: harikodali
"""
import os
import pickle
import pydload

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, dot, concatenate, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def get_text_encodings(texts, parameters):

    enc_seq = parameters["enc_token"].texts_to_sequences(texts)
    pad_seq = pad_sequences(enc_seq, maxlen=parameters["max_encoder_seq_length"],
                            padding='post')
    pad_seq = to_categorical(pad_seq, num_classes=parameters["enc_vocab_size"])
    return pad_seq


def get_extra_chars(parameters):
    allowed_extras = []
    for d_c, d_i in parameters["dec_token"].word_index.items():
        if d_c.lower() not in parameters["enc_token"].word_index:
            allowed_extras.append(d_i)
    return allowed_extras

def get_model_instance(parameters):

    encoder_inputs = Input(shape=(None, parameters["enc_vocab_size"],))
    encoder = Bidirectional(LSTM(128, return_sequences=True, return_state=True),
                            merge_mode='concat')
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    encoder_h = concatenate([forward_h, backward_h])
    encoder_c = concatenate([forward_c, backward_c])

    decoder_inputs = Input(shape=(None, parameters["dec_vocab_size"],))
    decoder_lstm = LSTM(256, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[encoder_h, encoder_c])

    attention = dot([decoder_outputs, encoder_outputs], axes=(2, 2))
    attention = Activation('softmax', name='attention')(attention)
    context = dot([attention, encoder_outputs], axes=(2, 1))
    decoder_combined_context = concatenate([context, decoder_outputs])

    output = TimeDistributed(Dense(128, activation="relu"))(decoder_combined_context)
    output = TimeDistributed(Dense(parameters["dec_vocab_size"], activation="softmax"))(output)

    model = Model([encoder_inputs, decoder_inputs], [output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def decode(model, parameters, input_texts, allowed_extras, batch_size):
    input_texts_c = input_texts.copy()
    out_dict = {}
    input_sequences = get_text_encodings(input_texts, parameters)

    parameters["reverse_dec_dict"][0] = "\n"
    outputs = [""]*len(input_sequences)

    target_text = "\t"
    target_seq = parameters["dec_token"].texts_to_sequences([target_text]*len(input_sequences))
    target_seq = pad_sequences(target_seq, maxlen=parameters["max_decoder_seq_length"],
                               padding="post")
    target_seq_hot = to_categorical(target_seq, num_classes=parameters["dec_vocab_size"])

    extra_char_count = [0]*len(input_texts)
    prev_char_index = [0]*len(input_texts)
    i = 0
    while len(input_texts) != 0:
        curr_char_index  = [i - extra_char_count[j] for j in range(len(input_texts))]
        input_encodings = np.argmax(input_sequences, axis=2)

        cur_inp_list = [input_encodings[_][curr_char_index[_]] if curr_char_index[_] < len(input_texts[_]) else 0 for _ in range(len(input_texts))]
        output_tokens = model.predict([input_sequences, target_seq_hot], batch_size=batch_size)
        sampled_possible_indices = np.argsort(output_tokens[:, i, :])[:, ::-1].tolist()
        sampled_token_indices = []
        for j, per_char_list in enumerate(sampled_possible_indices):
            for index in per_char_list:
                if index in allowed_extras:
                    if parameters["reverse_dec_dict"][index] == '\n' and cur_inp_list[j] != 0:
                        continue
                    elif parameters["reverse_dec_dict"][index] != '\n' and prev_char_index[j] in allowed_extras:
                        continue
                    sampled_token_indices.append(index)
                    extra_char_count[j] += 1
                    break
                elif parameters["enc_token"].word_index[parameters["reverse_dec_dict"][index].lower()] == cur_inp_list[j]:
                    sampled_token_indices.append(index)
                    break

        sampled_chars = [parameters["reverse_dec_dict"][index] for index in sampled_token_indices]

        outputs = [outputs[j] + sampled_chars[j] for j, output in enumerate(outputs)]
        end_indices = sorted([index for index, char  in enumerate(sampled_chars) if char == '\n'], reverse=True)
        for index in end_indices:
            out_dict[input_texts[index]] = outputs[index].strip()
            del  outputs[index]
            del input_texts[index]
            del extra_char_count[index]
            del sampled_token_indices[index]
            input_sequences = np.delete(input_sequences, index, axis=0)
            target_seq = np.delete(target_seq, index, axis=0)
        if i == parameters["max_decoder_seq_length"]-1 or len(input_texts) == 0:
            break
        target_seq[:,i+1] = sampled_token_indices
        target_seq_hot = to_categorical(target_seq, num_classes=parameters["dec_vocab_size"])
        prev_char_index = sampled_token_indices
        i += 1
    outputs = [out_dict[text] for text in input_texts_c]
    return outputs


model_links = {
            'en': {
                    'checkpoint': 'https://github.com/notAI-tech/fastPunct/releases/download/checkpoint-release/fastpunct_eng_weights.h5',
                    'params': 'https://github.com/notAI-tech/fastPunct/releases/download/checkpoint-release/parameter_dict.pkl'
                },
            
            }

lang_code_mapping = {
    'english': 'en',
    'french': 'fr',
    'italian': 'it'
}

class FastPunct():
    model = None
    parameters = None
    def __init__(self, lang_code="en", weights_path=None, params_path=None):
        if lang_code not in model_links and lang_code in lang_code_mapping:
            lang_code = lang_code_mapping[lang_code]
                
        if lang_code not in model_links:
            print("fastPunct doesn't support '" + lang_code + "' yet.")
            print("Please raise a issue at https://github.com/notai-tech/fastPunct/ to add this language into future checklist.")
            return None
        
        home = os.path.expanduser("~")
        lang_path = os.path.join(home, '.fastPunct_' + lang_code)
        if weights_path is None:
            weights_path = os.path.join(lang_path, 'checkpoint.h5')
        if params_path is None:
            params_path = os.path.join(lang_path, 'params.pkl')
        
        #if either of the paths are not mentioned, then, make lang directory from home
        if (params_path is None) or (weights_path is None):
            if not os.path.exists(lang_path):
                os.mkdir(lang_path)

        if not os.path.exists(weights_path):
            print('Downloading checkpoint', model_links[lang_code]['checkpoint'], 'to', weights_path)
            pydload.dload(url=model_links[lang_code]['checkpoint'], save_to_path=weights_path, max_time=None)

        if not os.path.exists(params_path):
            print('Downloading model params', model_links[lang_code]['params'], 'to', params_path)
            pydload.dload(url=model_links[lang_code]['params'], save_to_path=params_path, max_time=None)


        with open(params_path, "rb") as file:
            self.parameters = pickle.load(file)
        self.parameters["reverse_enc_dict"] = {i:c for c, i in self.parameters["enc_token"].word_index.items()}
        self.model = get_model_instance(self.parameters)
        self.model.load_weights(weights_path)
        self.allowed_extras = get_extra_chars(self.parameters)
    
    def punct(self, input_texts, batch_size=32):
        input_texts = [text.lower() for text in input_texts]
        return decode(self.model, self.parameters, input_texts, self.allowed_extras, batch_size)

    def fastpunct(self, input_texts, batch_size=32):
        # To be implemented
        return None
    
if __name__ == "__main__":
    fastpunct = FastPunct()
    print(fastpunct.punct(["oh i thought you were here", "in theory everyone knows what a comma is", "hey how are you doing", "my name is sheela i am in love with hrithik"]))
