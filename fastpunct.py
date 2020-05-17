# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:46:01 2020

@author: harikodali
"""

import pickle
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, dot, concatenate, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def get_text_encodings(texts, parameters):

    enc_seq = parameters["enc_token"].texts_to_sequences(texts)
    pad_seq = pad_sequences(enc_seq, maxlen=min(parameters["max_encoder_seq_length"], max([len(txt) for txt in texts])),
                            padding='post')
    pad_seq = to_categorical(pad_seq, num_classes=parameters["enc_vocab_size"])
    return pad_seq


def get_extra_chars(parameters):
    allowed_extras = []
    for d_c, d_i in parameters["dec_token"].word_index.items():
        if d_c.lower() not in parameters["enc_token"].word_index:
            allowed_extras.append(d_i)
    return allowed_extras + [0]

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
    i = 0
    while len(input_texts) != 0:
        curr_char_index  = [i - extra_char_count[j] for j in range(len(input_texts))]
        input_encodings = np.argmax(input_sequences, axis=2)
        cur_inp_list = [input_encodings[_][curr_char_index[_]] for _ in range(len(input_texts))]
        output_tokens = model.predict([input_sequences, target_seq_hot], batch_size=batch_size)
        sampled_possible_indices = np.argsort(output_tokens[:, i, :])[:, ::-1].tolist()
        sampled_token_indices = []
        for j, per_char_list in enumerate(sampled_possible_indices):
            for index in per_char_list:
                if index in allowed_extras:
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
        i += 1
    outputs = [out_dict[text] for text in input_texts_c]
    return outputs

class FastPunct():
    model = None
    parameters = None
    def __init__(self, params_path="parameter_dict.pkl", weights_path="fastpunct_eng_weights.h5"):
        with open(params_path, "rb") as file:
            self.parameters = pickle.load(file)
        self.model = get_model_instance(self.parameters)
        self.model.load_weights(weights_path)
        self.allowed_extras = get_extra_chars(self.parameters)
    
    def predict(self, input_texts, batch_size=512):
        return decode(self.model, self.parameters, input_texts, self.allowed_extras, batch_size)
    
    
if __name__ == "__main__":
    fastpunct = FastPunct()
    print(fastpunct.predict(["call haris mom", "oh i thought you were here", "where are you going", "in theory everyone knows what a comma is", "hey how are you doing", "my name is sheela i am in love with hrithik"]))
