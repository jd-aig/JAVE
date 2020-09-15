import sys
import tensorflow as tf
import numpy as np
import datetime

from utils import DataProcessor
from utils import EmbeddingContainer
from utils import load_vocabulary
from utils import compute_f1_score

paths = {
    "ckpt": "./ckpt/model.ckpt.batch1000",
    "vocab": "./vocab",
    "embedded": "./data/embedded",
    "test_data": "./data/test"
}

w2i_word, i2w_word = load_vocabulary(paths["vocab"] + "/vocab.word")
w2i_bio, i2w_bio = load_vocabulary(paths["vocab"] + "/vocab.bio")
w2i_label, i2w_label = load_vocabulary(paths["vocab"] + "/vocab.label")

embedding_container = EmbeddingContainer(
    paths["embedded"] + "/sids_of_txts",
    paths["embedded"] + "/txts.embedded.npy",
    paths["embedded"] + "/txts.embeddedG.npy",
    paths["embedded"] + "/cids_of_imgs",
    paths["embedded"] + "/imgs.embedded.npy",
    paths["embedded"] + "/imgs.embeddedG.npy"
)

data_processor = DataProcessor(
    paths["test_data"] + "/indexs",
    paths["test_data"] + "/input.seq",
    paths["test_data"] + "/output.seq",
    paths["test_data"] + "/output.label",
    w2i_word,
    w2i_bio, 
    w2i_label, 
    shuffling=False
)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

print("loading checkpoint from", paths["ckpt"], "...")

saver = tf.train.import_meta_graph(paths["ckpt"] + ".meta")
saver.restore(sess, paths["ckpt"])
graph = tf.get_default_graph()

# for n in graph.as_graph_def().node:
#     t = n.name
#     if not any(t.startswith(s) for s in ["opt/", "save/", "gradients/", "loss/", "Adam/"]):
#         print(t)

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
        graph.get_tensor_by_name("inputs_seq:0"): inputs_seq_batch,
        graph.get_tensor_by_name("inputs_seq_len:0"): inputs_seq_len_batch,
        graph.get_tensor_by_name("inputs_seq_embedded:0"): inputs_seq_embedded_batch,
        graph.get_tensor_by_name("inputs_seq_embeddedG:0"): inputs_seq_embeddedG_batch,
        graph.get_tensor_by_name("inputs_img_embedded:0"): inputs_img_embedded_batch,
        graph.get_tensor_by_name("inputs_img_embeddedG:0"): inputs_img_embeddedG_batch,
        graph.get_tensor_by_name("dropout_prob:0"): 0
    }

    preds_seq_batch, preds_label_batch = sess.run([
        graph.get_tensor_by_name("seq_projection/preds_seq:0"),
        graph.get_tensor_by_name("label_projection/preds_label:0")
    ], feed_dict)

    th = 0.5
    for pred_label in preds_label_batch:
        preds_attr.append([i2w_label[i] for i, v in enumerate(pred_label) if v > th])
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

print(result_attr)
print(result_value)
