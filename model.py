import tensorflow as tf

class MJAVE_Model(object):
    
    def __init__(self, 
                 txt_hidden_size, # hidden dim of pre-trained bert
                 img_hidden_size, # hidden dim of pre-trained resnet
                 img_global_size, # hidden dim of pre-trained resnet
                 img_block_num, # num of regional image features  (7×7=49)
                 attn_size, # hidden dim in attention
                 vocab_size_word, # vocab size of words
                 vocab_size_bio,  # vocab size of bio tokens
                 vocab_size_label,  # vocab size of attribute labels
                 use_labels, # whether use attribute prediction task to enhance value extraction task
                 use_KLloss, # whether use Kullback-Leibler loss to enhance value extraction task
                 use_images_global, # whether use global image features to enhance value extraction task
                 use_images_regional): # whether use regional image features to enhance value extraction task
        # inputs_seq: origin input word seq, shape=[B,S1], B=batch_size, S1=length_of_word_seq
        # self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq")
        # inputs_seq_len: lengths of input word seqs, shape=[B]
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len")
        # inputs_seq_embedded: text input encoded by pre-trained bert model(vectors of [CLS] and [SEP] have been removed), shape=[B,S1,D1], D1=hidden_dim_of_bert
        self.inputs_seq_embedded = tf.placeholder(tf.float32, [None, None, txt_hidden_size], name='inputs_seq_embedded')
        # inputs_seq_embeddedG: encoded vector of [CLS] by pre-trained bert model, shape=[B,D1]
        self.inputs_seq_embeddedG = tf.placeholder(tf.float32, [None, txt_hidden_size], name='inputs_seq_embeddedG')
        # inputs_img_embedded: regional image features encoded by pre-trained resnet model, shape=[B,S2,D2], S2=num_of_image_regions, D2=hidden_dim_of_resnet
        self.inputs_img_embedded = tf.placeholder(tf.float32, [None, img_block_num, img_hidden_size], name='inputs_img_embedded')
        # inputs_img_embeddedG: global image features encoded by pre-trained resnet model, shape=[B,D2]
        self.inputs_img_embeddedG = tf.placeholder(tf.float32, [None, img_global_size], name='inputs_img_embeddedG')
        # outputs_seq: output seq of bio tokens, shape=[B,S1]
        self.outputs_seq = tf.placeholder(tf.int32, [None, None], name='outputs_seq')
        # outputs_label: output of attribute labels, formed by 0 or 1, shape=[B,V], V=vocab_size_of_attribute_labels
        self.outputs_label = tf.placeholder(tf.float32, [None, vocab_size_label], name='outputs_label') # B * V
        # dropout_prob: probability of dropout layers
        self.dropout_prob = tf.placeholder(tf.float32, [], name="dropout_prob")
        
        input_seq_mask = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32) # B * S1
        contents_txt = self.inputs_seq_embedded * tf.expand_dims(input_seq_mask, axis=2) # B * S1 * D1
        contents_img = self.inputs_img_embedded # B * S2 * D2
        
        B = tf.shape(contents_txt)[0]
        S1 = tf.shape(contents_txt)[1]
        S2 = img_block_num
        
        with tf.variable_scope('txt_attn'):
            W_q_txt = tf.get_variable("W_q_txt", [txt_hidden_size, attn_size]) # D1 * D
            W_k_txt = tf.get_variable("W_k_txt", [txt_hidden_size, attn_size]) # D1 * D
            W_v_txt = tf.get_variable("W_v_txt", [txt_hidden_size, attn_size]) # D1 * D
            q_matrix = tf.reshape(tf.matmul(tf.reshape(contents_txt, [-1, txt_hidden_size]), W_q_txt), [-1, S1, attn_size]) # B * S1 * D 
            k_matrix = tf.reshape(tf.matmul(tf.reshape(contents_txt, [-1, txt_hidden_size]), W_k_txt), [-1, S1, attn_size]) # B * S1 * D
            v_matrix = tf.reshape(tf.matmul(tf.reshape(contents_txt, [-1, txt_hidden_size]), W_v_txt), [-1, S1, attn_size]) # B * S1 * D
            q_matrix = tf.expand_dims(q_matrix, 2) # B * S1 * 1 * D
            k_matrix = tf.expand_dims(k_matrix, 1) # B * 1 * S1 * D
            qk = tf.divide(tf.reduce_sum(q_matrix * k_matrix, axis=-1), tf.sqrt(float(attn_size))) # B * S1 * S1
            #qk = tf.nn.softmax(qk, axis=-1, name="txt_attn_score") # B * S1 * S1
            hiddens_txt = tf.matmul(qk, v_matrix) # B * S1 * D
    
        if use_images_global:
            with tf.variable_scope('img_attn'):
                W_q_img = tf.get_variable("W_q_img", [txt_hidden_size, attn_size]) # D1 * D
                W_k_img = tf.get_variable("W_k_img", [img_hidden_size, attn_size]) # D2 * D
                W_v_img = tf.get_variable("W_v_img", [img_hidden_size, attn_size]) # D2 * D
                q_matrix = tf.reshape(tf.matmul(tf.reshape(contents_txt, [-1, txt_hidden_size]), W_q_img), [-1, S1, attn_size]) # B * S1 * D 
                k_matrix = tf.reshape(tf.matmul(tf.reshape(contents_img, [-1, img_hidden_size]), W_k_img), [-1, S2, attn_size]) # B * S2 * D
                v_matrix = tf.reshape(tf.matmul(tf.reshape(contents_img, [-1, img_hidden_size]), W_v_img), [-1, S2, attn_size]) # B * S2 * D
                q_matrix = tf.expand_dims(q_matrix, 2) # B * S1 * 1 * D
                k_matrix = tf.expand_dims(k_matrix, 1) # B * 1 * S2 * D
                qk = tf.divide(tf.reduce_sum(q_matrix * k_matrix, axis=-1), tf.sqrt(float(attn_size))) # B * S1 * S2
                #qk = tf.nn.softmax(qk, axis=-1, name="img_attn_score") # B * S1 * S2
                mm_attn = qk
                mm_v = v_matrix

            with tf.variable_scope('global_gate'):
                d1 = tf.layers.dense(contents_txt, 1, use_bias=False) # B * S1 * 1
                d1 = tf.squeeze(d1, axis=-1) # B * S1
                d2 = tf.layers.dense(self.inputs_img_embeddedG, 1, use_bias=False) # B * 1
                b = tf.get_variable("b", [])
                g1 = tf.nn.sigmoid(d1 + d2 + b, name="gate_score") # B * S1
                hiddens_img1 = tf.expand_dims(g1, axis=2) * tf.matmul(mm_attn, mm_v) # B * S1 * D
        else:
            hiddens_img1 = tf.zeros([B, S1, attn_size])  # B * S1 * D
        
        hiddens_mm = hiddens_txt + hiddens_img1
        
        if use_labels:
             with tf.variable_scope('label_projection'):
                d3 = tf.layers.dense(tf.reduce_sum(contents_txt, axis=1), attn_size, use_bias=False) # B * D
                d3 = tf.layers.dropout(d3, 1-self.dropout_prob)
                d4 = tf.layers.dense(tf.reduce_sum(hiddens_mm, axis=1), attn_size, use_bias=False) # B * D
                d4 = tf.layers.dropout(d4, 1-self.dropout_prob)
                d5 = tf.layers.dense(self.inputs_seq_embeddedG, attn_size, use_bias=False) # B * D
                d5 = tf.layers.dropout(d5, 1-self.dropout_prob)
                logits_label = tf.layers.dense(d3 + d4 + d5, vocab_size_label) # B * V
                preds_label = tf.nn.sigmoid(logits_label, name="preds_label") # B * V
        else:
            preds_label = tf.zeros([B, vocab_size_label], name="preds_label") # B * V
        
        if use_images_regional and use_images_global and use_labels:
            with tf.variable_scope('regional_gate'):
                d6 = tf.layers.dense(preds_label, 1, use_bias=False) # B * 1
                d7 = tf.layers.dense(contents_img, 1, use_bias=False) # B * S2 * 1
                d7 = tf.squeeze(d7, axis=-1) # B * S2
                g2 = tf.nn.sigmoid(d6 + d7, name="gate_score") # B * S2
                hiddens_img2 = tf.matmul(mm_attn * tf.expand_dims(g2, axis=1), mm_v) # B * S1 * D
        else:
            hiddens_img2 = tf.zeros([B, S1, attn_size]) # B * S1 * D
        
        with tf.variable_scope('seq_projection'):
            d8 = tf.layers.dense(contents_txt, attn_size, use_bias=False) # B * S1 * D
            d8 = tf.layers.dropout(d8, 1-self.dropout_prob)
            d9 = tf.layers.dense(hiddens_mm, attn_size, use_bias=False) # B * S1 * D
            d9 = tf.layers.dropout(d9, 1-self.dropout_prob)
            d10 = tf.layers.dense(preds_label, attn_size, use_bias=False) # B * D
            d10 = tf.expand_dims(d10, axis=1) # B * 1 * D
            d10 = tf.layers.dropout(d10, 1-self.dropout_prob)
            logits_seq = tf.layers.dense(d8 + d9 + d10 + hiddens_img2, vocab_size_bio) # B * S1 * V
            preds_seq = tf.nn.softmax(logits_seq, name="preds_seq")
            
        self.outputs = [preds_seq, preds_label]
        
        with tf.variable_scope('loss'):
            loss_v = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq) # B * S1
            loss_v = tf.reduce_sum(loss_v * input_seq_mask, axis=-1) # B
            total_size = tf.reduce_sum(input_seq_mask, axis=1) # B
            loss_v = loss_v / total_size # B

            if use_labels:
                loss_a = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_label, labels=self.outputs_label) # B * V
                loss_a = tf.reduce_mean(loss_a, axis=-1) # B
            else:
                loss_a = 0
            
            if use_KLloss and use_labels:
                prob1 = preds_label[:, 2:] # B * (vocab_size_label - 2) (i.e. remove [PAD],[UNK])
                prob2 = tf.reduce_max(preds_seq, axis=1)[:, 3:] # B * (vocab_size_bio - 3) (i.e. remove [PAD],[UNK],O)
                index1 = [2*i for i in range(vocab_size_label-2)]
                index2 = [2*i+1 for i in range(vocab_size_label-2)]
                prob2_agg_b = tf.gather(prob2, [2*i for i in range(vocab_size_label-2)], axis=1)
                prob2_agg_i = tf.gather(prob2, [2*i+1 for i in range(vocab_size_label-2)], axis=1)
                prob2_agg = (prob2_agg_b + prob2_agg_i) / 2 # B * V
                loss_KL = self.relative_entropy(prob1, prob2_agg) # B * V
                loss_KL = tf.reduce_mean(loss_KL, axis=-1) # B
            else:
                loss_KL = 0
        
        r = 0.5
        total_loss = loss_v + loss_a + r * loss_KL # B * V
        self.loss = [
            tf.reduce_mean(loss_v), 
            tf.reduce_mean(loss_a), 
            tf.reduce_mean(loss_KL),
        ]
        
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        gradients = tf.gradients(total_loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = opt.apply_gradients(zip(clipped_gradients, params))
        
        print("model params:")
        params_num_all = 0
        for variable in tf.trainable_variables():
            params_num = 1
            for dim in variable.shape:
                params_num *= dim
            print("\t {} {}".format(variable.name, variable.shape))
            params_num_all += params_num
        print("all params num: " + str(params_num_all))
    
    def relative_entropy(self, p, q):
        p = tf.clip_by_value(p, 1e-8, 1.0)
        q = tf.clip_by_value(q, 1e-8, 1.0)
        return p * tf.log(p) - p * tf.log(q)

    
