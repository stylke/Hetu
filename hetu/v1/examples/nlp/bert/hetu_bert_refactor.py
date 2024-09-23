import hetu as ht
import numpy as np
import math

'''
Bert Module Architecture & Input/Output Tensor Size

BertModel Inputs: 
    input_ids: [batch_size, seq_len], word token indices in the vocabulary

BertModel Outputs:
    sequence_output: [batch_size, seq_len, hidden_size] (from BertEncoder)
    pooled_output: [batch_size, hidden_size] (from BertPooler)

BertModel:
    --[batch_size, seq_len]--
    BertEmbeddings:
        Embedding(word/position/token_type)
        LayerNorm
        Dropout
    --[batch_size, seq_len, hidden_size]--

    --[batch_size, seq_len, hidden_size]--
    BertEncoder:
        BertLayer(num_hidden_layers):
            BertAttention:
                BertSelfAttention
                --[batch_size, seq_len, hidden_size]--
                BertSelfOutput:
                    Linear
                    Dropout
                    Add & LayerNorm

            --[batch_size, seq_len, hidden_size]--
            BertIntermediate:
                Linear + Act(gule)
            --[batch_size, seq_len, intermediate_size]--
            BertOutput:
                Linear
                Dropout
                Add & LayerNorm
    --[batch_size, seq_len, hidden_size]--

    --[batch_size, seq_len, hidden_size]--
    BertPooler:
        (Slice, select [cls])
        --[batch_size, hidden_size]--
        Linear + Act(Tanh)
    --[batch_size, hidden_size]--

Bert
'''


'''
BertEmbeddings:
--------------------------------------------------------------------------------------------------'''


class BertEmbeddings(object):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        self.config = config
        self.seq_len = config.max_position_embeddings
        self.batch_size = config.batch_size

        self.word_embeddings = Embedding(
            config.vocab_size, config.hidden_size, "word_embeddings")
        self.position_embeddings = Embedding(
            config.max_position_embeddings, config.hidden_size, 'position_embeddings')
        self.token_type_embeddings = Embedding(
            config.type_vocab_size, config.hidden_size, 'token_type_embeddings')

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob)
        
    def to(self, dtype):
        self.word_embeddings.to(dtype)
        self.position_embeddings.to(dtype)
        self.token_type_embeddings.to(dtype)
        self.LayerNorm.to(dtype)

    def clone(self, dtype):
        model = BertEmbeddings(self.config)
        model.word_embeddings = self.word_embeddings.clone(dtype)
        model.position_embeddings = self.position_embeddings.clone(dtype)
        model.token_type_embeddings = self.token_type_embeddings.clone(dtype)
        model.LayerNorm = self.LayerNorm.clone(dtype)
        return model

    def __call__(self, input_ids, token_type_ids):
        '''
        inputs:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]

        outputs:
            embeddings: [batch_size, seq_len, hidden_size]
        '''
        seq_length = self.seq_len
        batch_size = self.batch_size
        position_ids = ht.from_numpy(np.arange(seq_length).reshape(
            (-1)).repeat(batch_size, axis=0).astype(np.int64), requires_grad=False)
        
        # ht.Variable('position_ids', value=np.arange(seq_length).reshape(
        #     (1, -1)).repeat(batch_size, axis=0), dtype=np.long, trainable=False, ctx=input_ids.ctx)

        '''Embedding Size
        inputs_id:[batch_size, seq_len], embedding_table:[vocab_size, hidden_size] 
        position_ids:[batch_size, seq_len], embedding_table:[seq_len, hidden_size]
        token_type_ids:[batch_size, seq_len], embedding_tabel:[type_vocab_size, hidden_size]
            --> embeddings: [batch_size, seq_len, hidden_size]
        '''
        words_embeddings = self.word_embeddings(input_ids)
        print("KKKKKK:",self.word_embeddings.weight)
        position_embeddings = self.position_embeddings(position_ids).reshape([batch_size, seq_length, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


'''-----------------------------------------------------------------------------------------------'''


'''
BertEncoder & BertLayer:
--------------------------------------------------------------------------------------------------'''


class BertEncoder(object):
    def __init__(self, config):
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.layer = [BertLayer(config)
                      for _ in range(config.num_hidden_layers)]

    def to(self, dtype):
        for layer in self.layer:
            layer.to(dtype)

    def clone(self, dtype):
        model = BertEncoder(self.config)
        model.layer = []
        for layer in self.layer:
            model.layer.append(layer.clone(dtype))
        return model

    def __call__(self, hidden_states, attention_mask=None):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            all_hidden_states: optional, num_hidden_layers * [batch_size, seq_len, hidden_size]
        '''

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        print("kkk:", hidden_states)
        return hidden_states  # last-layer hidden state


class BertLayer(object):
    def __init__(self, config):
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def to(self, dtype):
        self.attention.to(dtype)
        self.intermediate.to(dtype)
        self.output.to(dtype)
        
    def clone(self, dtype):
        model = BertLayer(self.config)
        model.attention = self.attention.clone(dtype)
        model.intermediate = self.intermediate.clone(dtype)
        model.output = self.output.clone(dtype)
        return model

    def __call__(self, hidden_states, attention_mask):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            layer_output: [batch_size, seq_len, hidden_size]
        '''
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


'''-----------------------------------------------------------------------------------------------'''


'''
BertAttention & BertSelfAttention & BertSelfOutput
--------------------------------------------------------------------------------------------------'''


class BertAttention(object):
    def __init__(self, config):
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def to(self, dtype):
        self.self.to(dtype)
        self.output.to(dtype)

    def clone(self, dtype):
        model = BertAttention(self.config)
        model.self = self.self.clone(dtype)
        model.output = self.output.clone(dtype)
        return model

    def __call__(self, input_tensor, attention_mask):
        '''
        inputs:
            input_tensor: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            attention_output: [batch_size, seq_len, hidden_size]
        '''
        print(input_tensor.shape)
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertSelfAttention(object):
    def __init__(self, config):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size  # all_head_size == hidden_size
        self.hidden_size = config.hidden_size
        self.seq_len = config.max_position_embeddings
        self.batch_size = config.batch_size
        self.config = config

        linear_input_shape = [self.batch_size, self.seq_len, self.hidden_size]
        self.query = Linear(config.hidden_size,
                            self.all_head_size, input_shape=linear_input_shape)
        self.key = Linear(config.hidden_size, self.all_head_size,
                          input_shape=linear_input_shape)
        self.value = Linear(config.hidden_size,
                            self.all_head_size, input_shape=linear_input_shape)

        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def to(self, dtype):
        self.query.to(dtype)
        self.key.to(dtype)
        self.value.to(dtype)

    def clone(self, dtype):
        model = BertSelfAttention(self.config)
        model.query = self.query.clone(dtype)
        model.key = self.key.clone(dtype)
        model.value = self.value.clone(dtype)
        return model

    def transpose_for_scores(self, input_tensor):
        output_tensor = ht.reshape(
            input_tensor, [self.batch_size, self.seq_len, self.num_attention_heads, self.attention_head_size])
        output_tensor = ht.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def transpose_key_for_scores(self, input_tensor):
        output_tensor = ht.reshape(
            input_tensor, [self.batch_size, self.seq_len, self.num_attention_heads, self.attention_head_size])
        output_tensor = ht.transpose(output_tensor, [0, 2, 3, 1])
        return output_tensor

    def __call__(self, hidden_states, attention_mask):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        outputs:
            context_layer: [batch_size, seq_len, hidden_size]
        '''

        # linear transformation
        # [batch_size, seq_len, hidden_size]
        mixed_query_layer = self.query(hidden_states)
        # [batch_size, seq_len, hidden_size]
        mixed_key_layer = self.key(hidden_states)
        # [batch_size, seq_len, hidden_size]
        mixed_value_layer = self.value(hidden_states)

        # transpose
        # [batch_size, num_heads, seq_len, head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # [batch_size, num_heads, head_size, seq_len]
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        # [batch_size, num_heads, seq_len, head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # score
        key_layer_scaled = key_layer * \
            (1.0 / np.sqrt(float(self.attention_head_size)))
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = ht.bmm(query_layer, key_layer_scaled)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores + \
            ht.broadcast(attention_mask, attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = ht.softmax(attention_scores, dim=3)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # [batch_size, num_heads, seq_len, head_size]
        context_layer = ht.bmm(attention_probs, value_layer)
        # [batch_size, seq_len, num_heads, head_size]
        context_layer = ht.transpose(context_layer, [0, 2, 1, 3])
        # [batch_size, seq_len, hidden_size]
        context_layer = ht.reshape(
            context_layer, [-1, self.seq_len, self.all_head_size])
        return context_layer


class BertSelfOutput(object):
    def __init__(self, config):
        self.config = config
        linear_input_shape = [config.batch_size,
                              config.max_position_embeddings, config.hidden_size]
        self.dense = Linear(config.hidden_size,
                            config.hidden_size, input_shape=linear_input_shape)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def to(self, dtype):
        self.dense.to(dtype)
        self.LayerNorm.to(dtype)

    def clone(self, dtype):
        model = BertSelfOutput(self.config)
        model.dense = self.dense.clone(dtype)
        model.LayerNorm = self.LayerNorm.clone(dtype)
        return model

    def __call__(self, hidden_states, input_tensor):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_tensor: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''-----------------------------------------------------------------------------------------------'''


'''
BertIntermediate & BertOutput （2-layer FeedForward)
--------------------------------------------------------------------------------------------------'''


class BertIntermediate(object):
    def __init__(self, config):
        self.config = config
        if config.hidden_act == "relu":
            self.intermediate_act_fn = ht.relu
        elif config.hidden_act == "gelu":
            assert("Not Implemented.")
            # self.intermediate_act_fn = ht.gelu_op
        linear_input_shape = [config.batch_size,
                              config.max_position_embeddings, config.hidden_size]
        self.dense = Linear(config.hidden_size, config.intermediate_size,
                            activation=self.intermediate_act_fn, input_shape=linear_input_shape)

    def to(self, dtype):
        self.dense.to(dtype)

    def clone(self, dtype):
        model = BertIntermediate(self.config)
        model.dense = self.dense.clone(dtype)
        return model

    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, intermediate_size]
        '''
        hidden_states = self.dense(hidden_states)
        return hidden_states


class BertOutput(object):
    def __init__(self, config):
        self.config = config
        linear_input_shape = [
            config.batch_size, config.max_position_embeddings, config.intermediate_size]
        self.dense = Linear(config.intermediate_size,
                            config.hidden_size, input_shape=linear_input_shape)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def to(self, dtype):
        self.dense.to(dtype)
        self.LayerNorm.to(dtype)

    def clone(self, dtype):
        model = BertOutput(self.config)
        model.dense = self.dense.clone(dtype)
        model.LayerNorm = self.LayerNorm.clone(dtype)
        return model

    def __call__(self, hidden_states, input_tensor):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, intermediate_size]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''-----------------------------------------------------------------------------------------------'''


'''
BertPooler
--------------------------------------------------------------------------------------------------'''


class BertPooler(object):
    def __init__(self, config):
        self.config = config
        self.dense = Linear(config.hidden_size,
                            config.hidden_size, activation=ht.tanh)
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size

    def to(self, dtype):
        self.dense.to(dtype)

    def clone(self, dtype):
        model = BertPooler(self.config)
        model.dense = self.dense.clone(dtype)
        return model

    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            pooled_output: [batch_size, hidden_size]
        '''
        first_token_tensor = ht.slice(
            hidden_states, [0, 0, 0], [self.batch_size, 1, self.hidden_size])
        first_token_tensor = ht.reshape(
            first_token_tensor, [self.batch_size, self.hidden_size])
        # pooled_output = first_token_tensor
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


'''-----------------------------------------------------------------------------------------------'''

'''
Bert Downstream Heads
--------------------------------------------------------------------------------------------------'''


class BertPredictionHeadTransform(object):
    def __init__(self, config):
        self.config = config
        if config.hidden_act == "relu":
            self.hidden_act = ht.relu
        elif config.hidden_act == "gelu":
            assert("Not Implemented.")
            # self.hidden_act = ht.gelu_op
        linear_input_shape = [config.batch_size,
                              config.max_position_embeddings, config.hidden_size]
        self.dense_act = Linear(config.hidden_size, config.hidden_size,
                                activation=self.hidden_act, input_shape=linear_input_shape)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def to(self, dtype):
        self.dense_act.to(dtype)
        self.LayerNorm.to(dtype)
        
    def clone(self, dtype):
        model = BertPredictionHeadTransform(self.config)
        model.dense_act = self.dense_act.clone(dtype)
        model.LayerNorm = self.LayerNorm.clone(dtype)
        return model

    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        '''
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(object):
    def __init__(self, config, bert_model_embedding_weights):
        '''
        bert_model_embedding_weights: [vocab_size, hidden_size]
        '''
        self.config = config
        self.bert_model_embedding_weights = bert_model_embedding_weights
        self.transform = BertPredictionHeadTransform(config)

        linear_input_shape = [config.batch_size,
                              config.max_position_embeddings, config.hidden_size]
        self.decoder = Linear(config.hidden_size, config.vocab_size,
                              bias_initializer=ht.nn.init.zeros_, input_shape=linear_input_shape)
        #self.decoder.weights = ht.transpose_op(bert_model_embedding_weights)
        self.decoder.weights = bert_model_embedding_weights

    def to(self, dtype):
        self.transform.to(dtype)
        self.decoder.to(dtype)
        
    def clone(self, dtype):
        model = BertLMPredictionHead(self.config, self.bert_model_embedding_weights)
        model.transform = self.transform.clone(dtype)
        model.decoder= self.decoder.clone(dtype)
        model.decoder.weights = self.bert_model_embedding_weights.to(datatype = dtype)
        return model

    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, vocab_size]
        '''
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(object):
    def __init__(self, config, bert_model_embedding_weights):
        self.config = config
        self.bert_model_embedding_weights = bert_model_embedding_weights
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def to(self, dtype):
        self.predictions.to(dtype)
    
    

    def __call__(self, sequence_output):
        '''
        inputs:
            sequence_output: [batch_size, seq_len, hidden_size]
        outputs:
            prediction_scores: [batch_size, seq_len, vocab_size]
        '''
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(object):
    def __init__(self, config):
        self.seq_relationship = Linear(config.hidden_size, 2)

    def to(self, dtype):
        self.seq_relationship.to(dtype)

    def __call__(self, pooled_output):
        '''
        inputs:
            pooled_output: [batch_size, hidden_size]
        outputs:
            seq_relationship_score: [batch_size, 2]
        '''
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(object):
    def __init__(self, config, bert_model_embedding_weights):
        self.config = config
        self.bert_model_embedding_weights = bert_model_embedding_weights
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = Linear(config.hidden_size, 2)

    def to(self, dtype):
        self.predictions.to(dtype)
        self.seq_relationship.to(dtype)

    def clone(self, dtype):
        model = BertPreTrainingHeads(self.config, self.bert_model_embedding_weights)
        model.predictions = self.predictions.clone(dtype)
        model.seq_relationship = self.seq_relationship.clone(dtype)
        return model

    def __call__(self, sequence_output, pooled_output):
        '''
        inputs:
            sequence_output: [batch_size, seq_len, hidden_size]
            pooled_output: [batch_size, hidden_size]
        outputs:
            prediction_scores: [batch_size, seq_len, vocab_size]
            seq_relationship_score: [batch_size, 2]
        '''
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


'''-----------------------------------------------------------------------------------------------'''


'''
BertModel:
--------------------------------------------------------------------------------------------------'''


class BertModel(object):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.batch_size = config.batch_size
        self.seq_len = config.max_position_embeddings
        self.config = config

    def to(self, dtype):
        self.embeddings.to(dtype)
        self.encoder.to(dtype)
        self.pooler.to(dtype)

    def clone(self, dtype):
        model = BertModel(self.config)
        model.embeddings = self.embeddings.clone(dtype)
        model.encoder = self.encoder.clone(dtype)
        model.pooler = self.pooler.clone(dtype)
        return model

    def __call__(self, input_ids, token_type_ids, attention_mask):
        extended_attention_mask = ht.reshape(
            attention_mask, [self.batch_size, 1, 1, self.seq_len])
        extended_attention_mask = (extended_attention_mask+(-1.0)) * 10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(
            embedding_output, extended_attention_mask)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output



'''-----------------------------------------------------------------------------------------------'''


'''
BertForPreTraining:
--------------------------------------------------------------------------------------------------'''


class BertForPreTraining(object):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        self.config = config
        self.bert = BertModel(config)
        index_all = ht.from_numpy(np.arange(config.vocab_size).astype(np.int64), requires_grad=False)
        
        # ht.Variable('index_all', value=np.arange(
        #     config.vocab_size), dtype=np.long, trainable=False)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings(index_all))
        # self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.vocab_size = config.vocab_size
    
    def to(self, dtype):
        self.bert.to(dtype)
        self.cls.to(dtype)

    def clone(self, dtype):
        model = BertForPreTraining(self.config)
        model.bert = self.bert.clone(dtype)
        model.cls = self.cls.clone(dtype)
        return model
        
        

    def __call__(self, input_ids, token_type_ids, attention_mask, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        return_op = [prediction_scores, seq_relationship_score]
        if masked_lm_labels is not None and next_sentence_label is not None:
            '''
            masked_lm_labels: [batch_size, seq_len]
            prediction_scores: [batch_size, seq_len, vocab_size]
            next_sentence_label: [batch_size]
            seq_relationship_score: [batch_size, 2]

            masked_lm_loss: [batch_size*seq_len]
            next_sentence_loss: [batch_size]
            '''

            # masked_lm_loss = ht.softmaxcrossentropy_sparse_op(prediction_scores, masked_lm_labels, ignored_index=-1)
            # next_sentence_loss = ht.softmaxcrossentropy_sparse_op(seq_relationship_score, next_sentence_label, ignored_index=-1)
            # masked_lm_loss = ht.crossentropy_sparse_op(ht.softmax_op(
            #     prediction_scores), masked_lm_labels, ignored_index=-1)
            # next_sentence_loss = ht.crossentropy_sparse_op(ht.softmax_op(
            #     seq_relationship_score), next_sentence_label, ignored_index=-1)
            masked_lm_loss = ht.softmax_cross_entropy_sparse(prediction_scores.reshape([-1, self.vocab_size]), masked_lm_labels.reshape([-1]), 
                                                             ignored_index = -1, reduction="sum")
            # masked_lm_loss = masked_lm_loss.sum()
            
            next_sentence_loss = ht.softmax_cross_entropy_sparse(seq_relationship_score, next_sentence_label, 
                                                                 ignored_index = -1)
            return_op += [masked_lm_loss, next_sentence_loss]
        return return_op


class BertForMaskedLM(object):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.vocab_size = config.vocab_size

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask)
        prediction_scores = self.cls(sequence_output)

        return_op = [prediction_scores]
        if masked_lm_labels is not None:
            '''
            masked_lm_labels: [batch_size, seq_len]
            prediction_scores: [batch_size, seq_len, vocab_size]

            masked_lm_loss: [batch_size*seq_len]
            '''
            # masked_lm_loss = ht.softmaxcrossentropy_sparse_op(prediction_scores, masked_lm_labels, ignored_index=-1)
            # masked_lm_loss = ht.crossentropy_sparse_op(ht.softmax_op(
            #     prediction_scores), masked_lm_labels, ignored_index=-1)
            masked_lm_loss = ht.softmax_cross_entropy_sparse(prediction_scores, masked_lm_labels, 
                                                             ignored_index = -1)
            return_op += [masked_lm_loss]

        return return_op


class BertForNextSentencePrediction(object):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        seq_relationship_score = self.cls(pooled_output)

        return_op = [seq_relationship_score]
        if next_sentence_label is not None:
            '''
            next_sentence_label: [batch_size]
            seq_relationship_score: [batch_size, 2]

            next_sentence_loss: [batch_size]
            '''
            # next_sentence_loss = ht.softmaxcrossentropy_sparse_op(seq_relationship_score, next_sentence_label, ignored_index=-1)
            next_sentence_loss = ht.softmax_cross_entropy_sparse(seq_relationship_score, next_sentence_label, 
                                                                 ignored_index = -1)
            # next_sentence_loss = ht.crossentropy_sparse_op(ht.softmax_op(
            #     seq_relationship_score), next_sentence_label, ignored_index=-1)
            return_op += [next_sentence_loss]

        return return_op


'''-----------------------------------------------------------------------------------------------'''


'''
Downstream tasks:
--------------------------------------------------------------------------------------------------'''


class BertForSequenceClassification(object):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]

        if labels is not None:
            # loss = ht.softmaxcrossentropy_sparse_op(logits, labels, ignored_index = -1)
            loss = ht.softmax_cross_entropy_sparse(logits, labels, ignored_index = -1)
            return loss, logits
        else:
            return logits


'''-----------------------------------------------------------------------------------------------'''


'''
Bert Layer utils (Embedding & BerLayerNorm & Dropout & Linear)
--------------------------------------------------------------------------------------------------'''


class Embedding(object):
    def __init__(self, num_embeddings, embedding_dim, embedding_name=None, initializer=ht.nn.init.normal_):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = ht.nn.functional.xavier_normal_([num_embeddings, embedding_dim], requires_grad=True)
        # ht.nn.Parameter(ht.empty([num_embeddings, embedding_dim], trainable=True))
        # ht.nn.init.xavier_normal_(self.weight)

    def to(self, dtype):
        self.weight = self.weight.to(datatype=dtype)

    def clone(self, dtype):
        model = Embedding(self.num_embeddings, self.embedding_dim)
        model.weight = self.weight.to(datatype=dtype)
        return model
    
    def __call__(self, input_tensor):
        return ht.embedding_lookup(self.weight, input_tensor)


class BertLayerNorm(object):
    def __init__(self, hidden_size, eps=1e-5):
        self.eps = eps
        self.hidden_size = hidden_size
        self.scale = ht.ones([hidden_size,], requires_grad=True)
        self.bias = ht.zeros([hidden_size,], requires_grad=True)

    def to(self, dtype):
        self.scale = self.scale.to(datatype=dtype)
        self.bias = self.bias.to(datatype=dtype)

    def clone(self, dtype):
        model = BertLayerNorm(self.hidden_size, self.eps)
        model.scale = self.scale.to(datatype=dtype)
        model.bias = self.bias.to(datatype=dtype)
        return model

    def __call__(self, x):
        # u = x.mean([-1], keepdims=[True])
        # s = (x - u)
        # s = s * s
        # s = s.mean([-1], keepdims=[True])
        # x = (x - u) / ht.sqrt(s + self.eps)
        # x = self.scale * x + self.bias
        # return x
        return ht.layer_norm(x, self.scale, self.bias, [768], self.eps)[0]


class Dropout(object):
    def __init__(self, dropout_prob=None):
        self.dropout_prob = dropout_prob

    def __call__(self, input_tensor):
        # if self.dropout_prob is None or self.dropout_prob == 0.0:
        #     return input_tensor
        # output = ht.dropout(input_tensor, 1.0 - self.dropout_prob)
        # return output
        return input_tensor


class Linear(object):
    def __init__(self, in_features, out_features, bias=True, activation=None, kernel_initializer=ht.nn.init.xavier_normal_, bias_initializer=ht.nn.init.zeros_, input_shape=None):
        self.bias_flag = bias
        self.activation = activation
        #self.weights = kernel_initializer(name='dense_weights', shape=(in_features, out_features))
        self.weights = ht.nn.functional.xavier_normal_([out_features, in_features], requires_grad=True)
        # self.weights = ht.nn.Parameter(ht.empty([out_features, in_features], trainable=True))
        # ht.nn.init.xavier_normal_(self.weights)
        if self.bias_flag:
            self.bias = ht.zeros([out_features,], requires_grad=True)
            # self.bias = ht.nn.Parameter(ht.zeros([out_features,], trainable=True))
            # ht.nn.init.zeros_(self.bias)
        self.input_shape = input_shape
        self.in_features = in_features
        self.out_features = out_features
        if self.input_shape is not None and self.input_shape[-1] != in_features:
            print("Specified in_features is not equal to input_shape[-1].")
            assert (False)

    def to(self, dtype):
        self.weights = self.weights.to(datatype=dtype)
        if self.bias_flag:
            self.bias = self.bias.to(datatype=dtype)

    def clone(self, dtype):
        model = Linear(self.in_features, self.out_features, self.bias_flag, self.activation, None, None, self.input_shape)
        model.weights = self.weights.to(datatype=dtype)
        if self.bias_flag:
            model.bias = self.bias.to(datatype=dtype)
        return model

    def __call__(self, input_tensor):
        if self.input_shape is not None and len(self.input_shape) != 2:
            input_tensor = ht.reshape(
                input_tensor, [-1, self.in_features])
        #outputs = ht.matmul_op(input_tensor, self.weights)
        outputs = ht.matmul(input_tensor, self.weights, trans_b=True)
        if self.bias_flag:
            outputs = outputs + ht.broadcast(self.bias, outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.input_shape is not None and len(self.input_shape) != 2:
            outputs = ht.reshape(
                outputs, self.input_shape[:-1]+[self.out_features])
        return outputs


'''-----------------------------------------------------------------------------------------------'''
