import os
import pickle 
import time 
from tqdm import tqdm
import argparse
import numpy as np 
import pickle
from bisect import bisect_left
from tensorflow.contrib import learn 
from tflearn.data_utils import to_categorical, pad_sequences

import tensorflow as tf

def main_logic(input_url):
    def get_word_vocab(urls, max_length_words, min_word_freq=0):
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length_words, min_frequency=min_word_freq) 
        start = time.time()
        x = np.array(list(vocab_processor.fit_transform(urls)))
        vocab_dict = vocab_processor.vocabulary_._mapping
        reverse_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
        return x, reverse_dict 

    def get_words(x, reverse_dict, delimit_mode, urls=None): 
        processed_x = []
        if delimit_mode == 0: 
            for url in x: 
                words = []
                for word_id in url: 
                    if word_id != 0: 
                        words.append(reverse_dict[word_id])
                    else: 
                        break
                processed_x.append(words) 
        elif delimit_mode == 1:
            for i in range(x.shape[0]):
                word_url = x[i]
                raw_url = urls[i]
                words = []
                for w in range(len(word_url)): 
                    word_id = word_url[w]
                    if word_id == 0: 
                        words.extend(list(raw_url))
                        break
                    else: 
                        word = reverse_dict[word_id]
                        idx = raw_url.index(word) 
                        special_chars = list(raw_url[0:idx])
                        words.extend(special_chars) 
                        words.append(word) 
                        raw_url = raw_url[idx+len(word):]
                        if w == len(word_url) - 1: 
                            words.extend(list(raw_url))
                processed_x.append(words)
        return processed_x 
    
    def get_char_ngrams(ngram_len, word): 
        word = "<" + word + ">" 
        chars = list(word) 
        begin_idx = 0
        ngrams = []
        while (begin_idx + ngram_len) <= len(chars): 
            end_idx = begin_idx + ngram_len 
            ngrams.append("".join(chars[begin_idx:end_idx])) 
            begin_idx += 1 
        return ngrams 
    
    def char_id_x(urls, char_dict, max_len_chars): 
        chared_id_x = []
        for url in urls: 
            url = list(url) 
            url_in_char_id = []
            l = min(len(url), max_len_chars)
            for i in range(l): 
                c = url[i] 
                try:
                    c_id = char_dict[c] 
                except KeyError:
                    c_id = 0
                url_in_char_id.append(c_id) 
            chared_id_x.append(url_in_char_id) 
        return chared_id_x 
        
    def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):   
        char_ngram_len = 1
        all_ngrams = set() 
        ngramed_x = []
        all_words = set() 
        worded_x = []
        for url in word_x: 
            url_in_ngrams = []
            url_in_words = []
            words = url
            for word in words:
                ngrams = get_char_ngrams(char_ngram_len, word) 
                if (len(ngrams) > max_len_subwords) or \
                    (high_freq_words is not None and len(word)>1 and not is_in(high_freq_words, word)):  
                    all_ngrams.update(ngrams[:max_len_subwords])
                    url_in_ngrams.append(ngrams[:max_len_subwords]) 
                    all_words.add("<UNKNOWN>")
                    url_in_words.append("<UNKNOWN>")
                else:     
                    all_ngrams.update(ngrams)
                    url_in_ngrams.append(ngrams) 
                    all_words.add(word) 
                    url_in_words.append(word) 
            ngramed_x.append(url_in_ngrams)
            worded_x.append(url_in_words) 
    
        all_ngrams = list(all_ngrams) 
        ngrams_dict = dict()
        for i in range(len(all_ngrams)):  
            ngrams_dict[all_ngrams[i]] = i+1
        all_words = list(all_words) 
        words_dict = dict() 
        for i in range(len(all_words)): 
            words_dict[all_words[i]] = i+1   
        ngramed_id_x = []
        for ngramed_url in ngramed_x: 
            url_in_ngrams = []
            for ngramed_word in ngramed_url: 
                ngram_ids = [ngrams_dict[x] for x in ngramed_word] 
                url_in_ngrams.append(ngram_ids) 
            ngramed_id_x.append(url_in_ngrams)  
        worded_id_x = []
        for worded_url in worded_x: 
            word_ids = [words_dict[x] for x in worded_url]
            worded_id_x.append(word_ids) 
        
        return ngramed_id_x, ngrams_dict, worded_id_x, words_dict 
    
    def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict = None): 
        char_ngram_len = 1
        ngramed_id_x = [] 
        worded_id_x = []
        if word_dict:
            word_vocab = sorted(list(word_dict.keys()))
        for url in word_x: 
            url_in_ngrams = [] 
            url_in_words = [] 
            words = url
            for word in words:
                ngrams = get_char_ngrams(char_ngram_len, word) 
                if len(ngrams) > max_len_subwords:
                    word = "<UNKNOWN>"  
                ngrams_id = [] 
                for ngram in ngrams: 
                    if ngram in ngram_dict: 
                        ngrams_id.append(ngram_dict[ngram]) 
                    else: 
                        ngrams_id.append(0) 
                url_in_ngrams.append(ngrams_id)
                if is_in(word_vocab, word): 
                    word_id = word_dict[word]
                else: 
                    word_id = word_dict["<UNKNOWN>"] 
                url_in_words.append(word_id)
            ngramed_id_x.append(url_in_ngrams)
            worded_id_x.append(url_in_words)
        
        return ngramed_id_x, worded_id_x 
    
    def is_in(a,x): 
        i = bisect_left(a,x)
        if i != len(a) and a[i] == x: 
            return True 
        else:
            return False 
    
    def pad_seq(urls, max_d1=0, max_d2=0, embedding_size=128): 
        if max_d1 == 0 and max_d2 == 0: 
            for url in urls: 
                if len(url) > max_d1: 
                    max_d1 = len(url) 
                for word in url: 
                    if len(word) > max_d2: 
                        max_d2 = len(word) 
        pad_idx = np.zeros((len(urls), max_d1, max_d2, embedding_size))
        pad_urls = np.zeros((len(urls), max_d1, max_d2))
        pad_vec = [1 for i in range(embedding_size)]
        for d0 in range(len(urls)): 
            url = urls[d0]
            for d1 in range(len(url)): 
                if d1 < max_d1: 
                    word = url[d1]
                    for d2 in range(len(word)): 
                        if d2 < max_d2: 
                            pad_urls[d0,d1,d2] = word[d2]
                            pad_idx[d0,d1,d2] = pad_vec
        return pad_urls, pad_idx
    
    def pad_seq_in_word(urls, max_d1=0, embedding_size=128):
        if max_d1 == 0: 
            url_lens = [len(url) for url in urls]
            max_d1 = max(url_lens)
        pad_urls = np.zeros((len(urls), max_d1))
        for d0 in range(len(urls)): 
            url = urls[d0]
            for d1 in range(len(url)): 
                if d1 < max_d1: 
                    pad_urls[d0,d1] = url[d1]
        return pad_urls 
    
    def softmax(x): 
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum() 
    
    def batch_iter(data, batch_size, num_epochs, shuffle=True): 
        data = np.array(data) 
        data_size = len(data) 
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 
        for epoch in range(num_epochs): 
            if shuffle: 
                shuffle_indices = np.random.permutation(np.arange(data_size)) 
                shuffled_data = data[shuffle_indices]
            else: 
                shuffled_data = data 
            for batch_num in range(num_batches_per_epoch): 
                start_idx = batch_num * batch_size 
                end_idx = min((batch_num+1) * batch_size, data_size)
                yield shuffled_data[start_idx:end_idx]
    
    def save_test_result(labels, all_predictions, all_scores): 
        output_labels = []
        for i in labels: 
            if i == 1: 
                output_labels.append(i) 
            else: 
                output_labels.append(-1) 
        output_preds = [] 
        for i in all_predictions: 
            if i == 1: 
                output_preds.append(i) 
            else: 
                output_preds.append(-1) 
        softmax_scores = [softmax(i) for i in all_scores]
        for i in range(len(output_labels)):
            output = str(int(output_labels[i])) + '\t' + str(int(output_preds[i])) + '\t' + str(softmax_scores[i][1]) + '\n'
            if softmax_scores[i][1] >= 0.75:
                return True, softmax_scores[i][1]
            else:
                return False, softmax_scores[i][1]

    def test_step(x, emb_mode):
        p = 1.0
        if emb_mode == 1: 
            feed_dict = {
                input_x_char_seq: x[0],
                dropout_keep_prob: p}  
        elif emb_mode == 2: 
            feed_dict = {
                input_x_word: x[0],
                dropout_keep_prob: p}
        elif emb_mode == 3: 
            feed_dict = {
                input_x_char_seq: x[0],
                input_x_word: x[1],
                dropout_keep_prob: p}
        elif emb_mode == 4: 
            feed_dict = {
                input_x_word: x[0],
                input_x_char: x[1],
                input_x_char_pad_idx: x[2],
                dropout_keep_prob: p}
        elif emb_mode == 5:  
            feed_dict = {
                input_x_char_seq: x[0],
                input_x_word: x[1],
                input_x_char: x[2],
                input_x_char_pad_idx: x[3],
                dropout_keep_prob: p}
        preds, s = sess.run([predictions, scores], feed_dict)
        return preds, s

    default_max_len_words = 100
    default_max_len_chars = 100
    default_max_len_subwords = 20
    default_delimit_mode = 1
    default_emb_dim = 32
    default_emb_mode = 5
    default_batch_size = 128
    urls = []
    labels = [ 1 ]

    urls.append(input_url)

    x, word_reverse_dict = get_word_vocab(urls, default_max_len_words) 
    word_x = get_words(x, word_reverse_dict, default_delimit_mode, urls) 
    ngram_dict = pickle.load(open("runs/10000/subwords_dict.p", "rb"))
    word_dict = pickle.load(open("runs/10000/words_dict.p", "rb"))
    ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, default_max_len_subwords, ngram_dict, word_dict) 
    chars_dict = pickle.load(open("runs/10000/chars_dict.p", "rb"))          
    chared_id_x = char_id_x(urls, chars_dict, default_max_len_chars)

    checkpoint_file = tf.train.latest_checkpoint("runs/10000/checkpoints/")
    graph = tf.Graph() 
    with graph.as_default(): 
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth=True 
        sess = tf.Session(config=session_conf)
        with sess.as_default(): 
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file) 
        
            if default_emb_mode in [1, 3, 5]: 
                input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
            if default_emb_mode in [2, 3, 4, 5]:
                input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
            if default_emb_mode in [4, 5]:
                input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
                input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0] 

            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]
         
            if default_emb_mode == 1: 
                batches = batch_iter(list(chared_id_x), default_batch_size, 1, shuffle=False) 
            elif default_emb_mode == 2: 
                batches = batch_iter(list(worded_id_x), default_batch_size, 1, shuffle=False) 
            elif default_emb_mode == 3: 
                batches = batch_iter(list(zip(chared_id_x, worded_id_x)), default_batch_size, 1, shuffle=False)
            elif default_emb_mode == 4: 
                batches = batch_iter(list(zip(ngramed_id_x, worded_id_x)), default_batch_size, 1, shuffle=False)
            elif default_emb_mode == 5: 
                batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), default_batch_size, 1, shuffle=False)    
            all_predictions = []
            all_scores = []
        
            nb_batches = int(len(labels) / default_batch_size)
            if len(labels) % default_batch_size != 0: 
              nb_batches += 1 
            it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} test_size {}".format(default_emb_mode, default_delimit_mode, len(labels)), ncols=0)
            for idx in it:
                batch = next(batches)

                if default_emb_mode == 1: 
                    x_char_seq = batch 
                elif default_emb_mode == 2: 
                    x_word = batch 
                elif default_emb_mode == 3: 
                    x_char_seq, x_word = zip(*batch) 
                elif default_emb_mode == 4: 
                    x_char, x_word = zip(*batch)
                elif default_emb_mode == 5: 
                    x_char, x_word, x_char_seq = zip(*batch)        

                x_batch = []    
                if default_emb_mode in[1, 3, 5]: 
                    x_char_seq = pad_seq_in_word(x_char_seq, default_max_len_chars) 
                    x_batch.append(x_char_seq)
                if default_emb_mode in [2, 3, 4, 5]:
                    x_word = pad_seq_in_word(x_word, default_max_len_words) 
                    x_batch.append(x_word)
                if default_emb_mode in [4, 5]:
                    x_char, x_char_pad_idx = pad_seq(x_char, default_max_len_words, default_max_len_subwords, default_emb_dim)
                    x_batch.extend([x_char, x_char_pad_idx])
            
                batch_predictions, batch_scores = test_step(x_batch, default_emb_mode)            
                all_predictions = np.concatenate([all_predictions, batch_predictions]) 
                all_scores.extend(batch_scores) 

                it.set_postfix()

    res = save_test_result(labels, all_predictions, all_scores)
    return res
