import sys
import tensorflow as tf
import logging
import numpy as np
import datetime

from model import MJAVE_Model
from utils import DataProcessor
from utils import EmbeddingContainer
from utils import load_vocabulary
from utils import compute_f1_score

config = {
    "txt_hidden_size": 768, # hidden dim of pre-trained bert
    "img_hidden_size": 2048, # hidden dim of pre-trained resnet(last conv layer)
    "img_global_size": 2048, # hidden dim of pre-trained resnet(last pooling layer)
    "img_block_num": 49, # # num of regional image features  (7×7=49)
    "attn_size": 200, # hidden dim in attention
    "batch_size": 128, # batch size
    "dropout_prob": 0 # probability of dropout layers
}

paths = {
    "ckpt": "./ckpt/model.ckpt",
    "vocab": "./vocab",
    "embedded": "./data/embedded",
    "train_data": "./data/train",
    "valid_data": "./data/valid",
    "test_data": "./data/test"
}

w2i_word, i2w_word = load_vocabulary(paths["vocab"] + "/vocab.word")
w2i_bio, i2w_bio = load_vocabulary(paths["vocab"] + "/vocab.bio")
w2i_label, i2w_label = load_vocabulary(paths["vocab"] + "/vocab.label")

# embedding_container: restore all vectors encoded by pre-trained bert and resnet
embedding_container = EmbeddingContainer(
    paths["embedded"] + "/sids_of_txts", # indexes to find text encoded vector 
    paths["embedded"] + "/txts.embedded.npy", # text encoded by pre-trained bert, shape=[N, max_len_of_word_seqs, dim_of_bert_output]
    paths["embedded"] + "/txts.embeddedG.npy", # vectors of [CLS] encoded by a pre-trained bert, shape=[N, dim_of_bert_output]
    paths["embedded"] + "/cids_of_imgs", # indexes to find image encoded vector 
    paths["embedded"] + "/imgs.embedded.npy", # image encoded by pre-trained resnet, shape=[N, image_region_num, dim_of_resnet_output]
    paths["embedded"] + "/imgs.embeddedG.npy" # image encoded by pre-trained resnet, shape=[N, dim_of_resnet_output]
)

# data_processor: utils for data processing(load data, get batch samples, etc.) 
data_processor_train = DataProcessor(
    paths["train_data"] + "/indexs",
    paths["train_data"] + "/input.seq",
    paths["train_data"] + "/output.seq",
    paths["train_data"] + "/output.label",
    w2i_word,
    w2i_bio, 
    w2i_label, 
    shuffling=True
)

data_processor_valid = DataProcessor(
    paths["valid_data"] + "/indexs",
    paths["valid_data"] + "/input.seq",
    paths["valid_data"] + "/output.seq",
    paths["valid_data"] + "/output.label",
    w2i_word,
    w2i_bio, 
    w2i_label,  
    shuffling=False
)

data_processor_test = DataProcessor(
    paths["test_data"] + "/indexs",
    paths["test_data"] + "/input.seq",
    paths["test_data"] + "/output.seq",
    paths["test_data"] + "/output.label",
    w2i_word,
    w2i_bio, 
    w2i_label, 
    shuffling=False
)

use_labels = True # whether use attribute prediction task to enhance value extraction task
use_KLloss = True # whether use Kullback-Leibler loss to enhance value extraction task
use_images_global = True # whether use global image features to enhance value extraction task
use_images_regional = True # whether use regional image features to enhance value extraction task

# build model
model = MJAVE_Model(
    config["txt_hidden_size"], 
    config["img_hidden_size"], 
    config["img_global_size"], 
    config["img_block_num"], 
    config["attn_size"],
    len(w2i_word),
    len(w2i_bio),
    len(w2i_label),
    use_labels, 
    use_KLloss, 
    use_images_global, 
    use_images_regional
)

# start training
saver = saver = tf.train.Saver(max_to_keep=10)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    
    epoches = 0
    losses = [0, 0, 0]
    batches = 0
    val_best_f1 = 0

    while epoches < 50:
        (inputs_seq_batch, 
         inputs_seq_len_batch, 
         inputs_seq_embedded_batch, 
         inputs_seq_embeddedG_batch, 
         inputs_img_embedded_batch, 
         inputs_img_embeddedG_batch, 
         outputs_seq_batch, 
         outputs_label_batch) = data_processor_train.get_batch(config["batch_size"], embedding_container)
               
        feed_dict = {
            model.inputs_seq_len: inputs_seq_len_batch,
            model.inputs_seq_embedded: inputs_seq_embedded_batch,
            model.inputs_seq_embeddedG: inputs_seq_embeddedG_batch,
            model.inputs_img_embedded: inputs_img_embedded_batch,
            model.inputs_img_embeddedG: inputs_img_embeddedG_batch,
            model.outputs_seq: outputs_seq_batch,
            model.outputs_label: outputs_label_batch,
            model.dropout_prob: config["dropout_prob"]
        }
        
        # preview the data of the first batch
        if batches == 0: 
            print("###### shape of a batch #######")
            print("inputs_seq:", inputs_seq_batch.shape)
            print("inputs_seq_len:", inputs_seq_len_batch.shape)
            print("inputs_seq_embedded:", inputs_seq_embedded_batch.shape)
            print("inputs_seq_embeddedG:", inputs_seq_embeddedG_batch.shape)
            print("inputs_img_embedded:", inputs_img_embedded_batch.shape)
            print("inputs_img_embeddedG:", inputs_img_embeddedG_batch.shape)
            print("outputs_seq:", outputs_seq_batch.shape)
            print("outputs_label:", outputs_label_batch.shape)
            print("###### preview a sample #######")
            print("inputs_seq:", " ".join([i2w_word[i] for i in inputs_seq_batch[0]]))
            print("inputs_seq_len:", inputs_seq_len_batch[0])
            print("outputs_seq:", " ".join([i2w_bio[i] for i in outputs_seq_batch[0]]))
            print("outputs_label:", " ".join([i2w_label[i] for i, v in enumerate(outputs_label_batch[0]) if v == 1]))
            print("###############################")
        
        loss, _ = sess.run([model.loss, model.train_op], feed_dict)
        for i in range(3):
            losses[i] += loss[i]
        batches += 1
                
        if data_processor_train.end_flag:
            data_processor_train.refresh()
            epoches += 1
        
        # evaluate on valid dataset
        def valid(data_processor):
            preds_attr = []
            golds_attr = []
            preds_bio = []
            golds_bio = []

            while True:
                (inputs_seq_batch, 
                 inputs_seq_len_batch, 
                 inputs_seq_embedded_batch, 
                 inputs_seq_embeddedG_batch, 
                 inputs_img_embedded_batch, 
                 inputs_img_embeddedG_batch, 
                 outputs_seq_batch, 
                 outputs_label_batch) = data_processor.get_batch(512, embedding_container)

                feed_dict = {
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.inputs_seq_embedded: inputs_seq_embedded_batch,
                    model.inputs_seq_embeddedG: inputs_seq_embeddedG_batch,
                    model.inputs_img_embedded: inputs_img_embedded_batch,
                    model.inputs_img_embeddedG: inputs_img_embeddedG_batch,
                    model.dropout_prob: 0
                }
                
                preds_seq_batch, preds_label_batch = sess.run(model.outputs, feed_dict)
                
                if use_labels:
                    th = 0.5
                    for pred_label in preds_label_batch:
                        preds_attr.append([i2w_label[i] for i, v in enumerate(pred_label) if v > th])
                else:
                    for pred_seq in preds_seq_batch:
                        pred_attr = []
                        for i in np.argmax(pred_seq, -1):
                            w = i2w_bio[i]
                            if "-" in w:
                                pred_attr.append(w.split("-")[1])
                        preds_attr.append(list(set(pred_attr)))

                for gold_label in outputs_label_batch:
                    golds_attr.append([i2w_label[i] for i, v in enumerate(gold_label) if v == 1])

                for pred_seq, gold_seq, l in zip(preds_seq_batch, outputs_seq_batch, inputs_seq_len_batch):
                    pred_seq = np.argmax(pred_seq, -1)
                    preds_bio.append([i2w_bio[i] for i in pred_seq[:l]])
                    golds_bio.append([i2w_bio[i] for i in gold_seq[:l]])

                if data_processor.end_flag:
                    data_processor.refresh()
                    break

            p_sum = 0
            r_sum = 0
            hits = 0
            for pred_attr, gold_attr in zip(preds_attr, golds_attr):
                p_sum += len(pred_attr)
                r_sum += len(gold_attr)
                for a in pred_attr:
                    if a in gold_attr:
                        hits += 1
            p = hits*100 / p_sum if p_sum != 0 else 0
            r = hits*100 / r_sum if r_sum != 0 else 0
            f1 = (2 * p * r) / (p + r) if p + r > 0 else 0
            result_attr = [p, r, f1]

            p, r, f1 = compute_f1_score(golds_bio, preds_bio)
            result_value = [p, r, f1]

            return result_value, result_attr
        
        batches_for_print = 1  
        if batches % batches_for_print == 0:
            logging.info("")
            logging.info("Epoches: " + str(epoches))
            logging.info("Batches: " + str(batches))
            logging.info("Loss of Value: " + str(losses[0]/batches_for_print))
            logging.info("Loss of Attr: " + str(losses[1]/batches_for_print))
            logging.info("Loss of KL: " + str(losses[2]/batches_for_print))
            losses = [0, 0, 0]

            (p1, r1, f11), (p2, r2, f12) = valid(data_processor_valid)
            logging.info("Valid Attr P/R/F1: {} / {} / {}".format(round(p2, 2), round(r2, 2), round(f12, 2)))
            logging.info("Valid Value P/R/F1: {} / {} / {}".format(round(p1, 2), round(r1, 2), round(f11, 2)))
            
            if f11 > val_best_f1:
                logging.info("################# best performance now ###################")
                val_best_f1 = f11
            
                (p1, r1, f11), (p2, r2, f12) = valid(data_processor_test)
                logging.info("Test Attr P/R/F1: {} / {} / {}".format(round(p2, 2), round(r2, 2), round(f12, 2)))
                logging.info("Test Value P/R/F1: {} / {} / {}".format(round(p1, 2), round(r1, 2), round(f11, 2)))
                
                saver.save(sess, paths["ckpt"] + ".batch{}".format(batches))
            
            

            

