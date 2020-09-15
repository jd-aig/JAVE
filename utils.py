import numpy as np
import random

##########################
####### Vocabulary #######
##########################
            
def load_vocabulary(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = f.read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w

##################################
####### EmbeddingContainer #######
##################################

class EmbeddingContainer(object):
    def __init__(self, path1, path2, path3, path4, path5, path6):
        # load vectors of text processed by pre-trained BERT
        self.sid2txtindex = {line: i for i, line in enumerate(open(path1, "r").read().strip().split("\n"))}
        self.txts_embedded = np.load(path2)
        self.txts_embeddedG = np.load(path3)
        # load vectors of image processed by pre-trained Resnet
        self.cid2imgindex = {line: i for i, line in enumerate(open(path4, "r").read().strip().split("\n"))}
        self.imgs_embedded = np.load(path5)
        self.imgs_embeddedG = np.load(path6)
        print("EmbeddingContainer load embeddings:")
        print("\t txts_embedded: {} {}".format(self.txts_embedded.shape, self.txts_embedded.dtype))
        print("\t txts_embeddedG: {} {}".format(self.txts_embeddedG.shape, self.txts_embeddedG.dtype))
        print("\t imgs_embedded: {} {}".format(self.imgs_embedded.shape, self.imgs_embedded.dtype))
        print("\t imgs_embeddedG: {} {}".format(self.imgs_embeddedG.shape, self.imgs_embeddedG.dtype))
        
    def get_txt_embedded_vector(self, sid):
        index = self.sid2txtindex[sid]
        return self.txts_embedded[index], self.txts_embeddedG[index]
    
    def get_img_embedded_vector(self, cid):
        index = self.cid2imgindex[cid]
        return self.imgs_embedded[index], self.imgs_embeddedG[index]
    
    
#############################
####### DataProcessor #######
#############################
    
class DataProcessor(object):
    def __init__(self, path1, path2, path3, path4, w2i_word, w2i_bio, w2i_label, shuffling=False):
        
        cids = [] # content index (image index) of inputs
        sids = [] # sentence index (text index) of inputs
        for line in open(path1).read().strip().split("\n"):
            cids.append(line.split("\t")[0])
            sids.append(line.split("\t")[1])
        
        # load inputs
        inputs_seq = []
        with open(path2, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                inputs_seq.append([w2i_word[w] if w in w2i_word else w2i_word["[UNK]"] for w in line.split(" ")])
        
        outputs_seq = []
        with open(path3, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                outputs_seq.append([w2i_bio[w] for w in line.split(" ")])
        
        outputs_label = []
        with open(path4, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                output_label = [0] * len(w2i_label)
                for w in line.split(" "):
                    if w != "[PAD]":
                        index = w2i_label[w]
                        output_label[index] = 1
                outputs_label.append(output_label)
        
        self.sids = sids
        self.cids = cids
        self.w2i_word = w2i_word
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.outputs_label = outputs_label
        self.shuffling = shuffling
        self.ps = list(range(len(inputs_seq)))
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)))
    
    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
    
    def get_batch(self, batch_size, embedding_container):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        inputs_seq_embedded_batch = []
        inputs_seq_embeddedG_batch = []
        inputs_img_embedded_batch = []
        inputs_img_embeddedG_batch = []
        outputs_seq_batch = []
        outputs_label_batch = []
        
        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            outputs_label_batch.append(self.outputs_label[p].copy())
            seq_embedded, seq_embeddedG = embedding_container.get_txt_embedded_vector(self.sids[p])
            img_embedded, img_embeddedG = embedding_container.get_img_embedded_vector(self.cids[p])
            inputs_seq_embedded_batch.append(seq_embedded)
            inputs_seq_embeddedG_batch.append(seq_embeddedG)
            inputs_img_embedded_batch.append(img_embedded)
            inputs_img_embeddedG_batch.append(img_embeddedG)
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True
        
        max_seq_len = max(inputs_seq_len_batch)
        
        for i, l in enumerate(inputs_seq_len_batch):
            inputs_seq_batch[i].extend([self.w2i_word["[PAD]"]] * (max_seq_len - l))
            outputs_seq_batch[i].extend([self.w2i_bio["O"]] * (max_seq_len - l))
            inputs_seq_embedded_batch[i] = inputs_seq_embedded_batch[i][:max_seq_len,:]
            
        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(inputs_seq_embedded_batch, dtype="float32"),
                np.array(inputs_seq_embeddedG_batch, dtype="float32"),
                np.array(inputs_img_embedded_batch, dtype="float32"),
                np.array(inputs_img_embeddedG_batch, dtype="float32"),
                np.array(outputs_seq_batch, dtype="int32"),
                np.array(outputs_label_batch, dtype="float32"))

###########################################################    
####### compute f1 score is modified from SlotGated #######
#######   https://github.com/MiuLab/SlotGated-SLU   #######
###########################################################  

def __startOfChunk(prevTag, tag, prevTagType, tagType):
    if prevTag == 'B' and tag == 'B':
        return True
    if prevTag == 'I' and tag == 'B':
        return True
    if prevTag == 'O' and tag == 'B':
        return True
    if prevTag == 'O' and tag == 'I':
        return True
#     if prevTag == 'E' and tag == 'E':
#         return True
#     if prevTag == 'E' and tag == 'I':
#         return True
#     if prevTag == 'O' and tag == 'E':
#         return True
    if tag != 'O' and prevTagType != tagType:
        return True
    return False

def __endOfChunk(prevTag, tag, prevTagType, tagType):
    if prevTag == 'B' and tag == 'B':
        return True
    if prevTag == 'B' and tag == 'O':
        return True
    if prevTag == 'I' and tag == 'B':
        return True
    if prevTag == 'I' and tag == 'O':
        return True
#     if prevTag == 'E' and tag == 'E':
#         return True
#     if prevTag == 'E' and tag == 'I':
#         return True
#     if prevTag == 'E' and tag == 'O':
#         return True
#     if prevTag == 'I' and tag == 'O':
#         return True
    if prevTag != 'O' and prevTagType != tagType:
        return True
    return False

def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType

def compute_f1_score(gold_slots, pred_slots):
    correctChunkCnt = 0
    goldChunkCnt = 0
    predChunkCnt = 0
    for gold_slot, pred_slot in zip(gold_slots, pred_slots):
        in_correcting = False
        lastGoldTag = 'O'
        lastGoldType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(gold_slot, pred_slot):
            goldTag, goldType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if in_correcting == True:
                if __endOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastGoldType == lastPredType):
                    in_correcting = False
                    correctChunkCnt += 1
                elif __endOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) != \
                     __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                     (goldType != predType):
                    in_correcting = False

            if __startOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (goldType == predType):
                in_correcting = True

            if __startOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) == True:
                goldChunkCnt += 1
                
            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                predChunkCnt += 1
                
            lastGoldTag = goldTag
            lastGoldType = goldType
            lastPredTag = predTag
            lastPredType = predType

        if in_correcting == True:
            correctChunkCnt += 1
            
    if predChunkCnt > 0:
        precision = 100*correctChunkCnt/predChunkCnt
    else:
        precision = 0

    if goldChunkCnt > 0:
        recall = 100*correctChunkCnt/goldChunkCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return precision, recall, f1
