# The classes used for data processing and convert_examples_to_features are very similar versions of the ones \
# found in Hugging Face's scripts in the transformers library. For more BERT or similar language model implementation \
# examples, we would highly recommend checking that library as well.


from __future__ import absolute_import, division, print_function

import csv

import numpy as np
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Classes regarding input and data handling

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree


class InputFeatures(object):
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.agree = agree


class DataProcessor(object):
    """Base class to read data files."""

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        return lines


class FinSentProcessor(DataProcessor):
    """
    Data processor for FinBERT.
    """

    def get_examples(self, data_dir, phase):
        """
        Get examples from the data directory.

        Parameters
        ----------
        data_dir: str
            Path for the data directory.
        phase: str
            Name of the .csv file to be loaded.
        """
        return self._create_examples(self._read_tsv(os.path.join(data_dir, (phase + ".csv"))), phase)

    def get_labels(self):
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text = line[1]
            label = line[2]
            try:
                agree = line[3]
            except:
                agree = None
            examples.append(
                InputExample(guid=guid, text=text, label=label, agree=agree))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode='classification'):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.

    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'

    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                              len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode == 'classification':
            label_id = label_map[example.label]
        elif mode == 'regression':
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          agree=agree))
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


def get_metrics(df):
    "Computes accuracy and precision-recall for different sentiments."

    df.loc[:, 'guess'] = df.predictions.apply(np.argmax)
    df.loc[:, 'accurate'] = df.apply(lambda x: x['guess'] == x['labels'], axis=1)
    accuracy = df.accurate.sum() / df.shape[0]

    pos_recall = df[df['labels'] == 0].accurate.sum() / df[df['labels'] == 0].shape[0]
    neg_recall = df[df['labels'] == 1].accurate.sum() / df[df['labels'] == 1].shape[0]
    net_recall = df[df['labels'] == 2].accurate.sum() / df[df['labels'] == 2].shape[0]

    pos_precision = df[df['guess'] == 0].accurate.sum() / df[df['guess'] == 0].shape[0]
    neg_precision = df[df['guess'] == 1].accurate.sum() / df[df['guess'] == 1].shape[0]
    net_precision = df[df['guess'] == 2].accurate.sum() / df[df['guess'] == 2].shape[0]

    pos_f1score = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    neg_f1score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    net_f1score = 2 * (net_precision * net_recall) / (net_precision + net_recall)

    return {'Accuracy': accuracy,
            'Positive': {'precision': pos_precision, 'recall': pos_recall, 'f1-score': pos_f1score}, 'Negative': \
                {'precision': neg_precision, 'recall': neg_recall, 'f1-score': neg_f1score},
            'Neutral': {'precision': net_precision, 'recall': net_recall, 'f1-score': net_f1score}}


def get_prediction(text, model, tokenizer, max_seq_length = 512):
    """
    Get one prediction.

    Parameters
    ----------
    text: str
        The text to be analyzed.
    model: BertModel
        The model to be used.
    tokenizer: BertTokenizer
        The tokenizer to be used.

    Returns
    -------
    predition: np.array
        An array that includes probabilities for each class.
    """

    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                              len(tokens) - (3 * max_seq_length // 4) + 1:]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (64 - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    features = []
    features.append(
        InputFeatures(input_ids=input_ids,
                      segment_ids=segment_ids,
                      input_mask=input_mask,
                      label_id=None))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    model.eval()
    prediction = softmax(model(all_input_ids, all_segment_ids, all_input_mask).detach().numpy())
    return prediction

def chunks(l, n):
    """
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

####################################################################################################
#################### FINBERT########################################################################
#################################################################################################### 

import random

import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Config(object):
    """The configuration class for training."""

    def __init__(self,
                 data_dir,
                 bert_model,
                 model_dir,
                 max_seq_length=64,
                 train_batch_size=32,
                 eval_batch_size=32,
                 learning_rate=5e-5,
                 num_train_epochs=10.0,
                 warm_up_proportion=0.1,
                 no_cuda=False,
                 do_lower_case=True,
                 seed=42,
                 local_rank=-1,
                 gradient_accumulation_steps=1,
                 fp16=False,
                 output_mode='classification',
                 discriminate=True,
                 gradual_unfreeze=True,
                 encoder_no=12):
        """

        Parameters
        ----------
        data_dir: str
            Path for the training and evaluation datasets.
        bert_model: BertModel
            The BERT model to be used. For example: BertForSequenceClassification.from_pretrained(...)
        model_dir: str
            The path where the resulting model will be saved.
        max_seq_length: int
            The maximum length of the sequence to be used. Default value is 64.
        train_batch_size: int
            The batch size for the training. Default value is 32.
        eval_batch_size: int
            The batch size for the evaluation. Default value is 32.
        learning_rate: float
            The learning rate. Default value is 5e5.
        num_train_epochs: int
            Number of epochs to train. Default value is 4.
        warm_up_proportion: float
            During the training, the learning rate is linearly increased. This value determines when the learning rate
            reaches the intended learning rate. Default value is 0.1.
        no_cuda: bool
            Determines whether to use gpu. Default is False.
        do_lower_case: bool
            Determines whether to make all training and evaluation examples lower case. Default is True.
        seed: int
            Random seed. Defaults to 42.
        local_rank: int
            Used for number of gpu's that will be utilized. If set -1, no distributed training will be done. Default
            value is -1.
        gradient_accumulation_steps: int
            Number of gradient accumulations steps. Defaults to 1.
        fp16: bool
            Determines whether to use 16 bits for floats, instead of 32.
        output_mode: 'classification' or 'regression'
            Determines whether the task is classification or regression.
        discriminate: bool
            Determines whether to apply discriminative fine-tuning. 
        gradual_unfreeze: bool
            Determines whether to gradually unfreeze lower and lower layers as the training goes on.
        encoder_no: int
            Starting from which layer the model is going to be finetuned. If set 12, whole model is going to be
            fine-tuned. If set, for example, 6, only the last 6 layers will be fine-tuned.
        """
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.local_rank = local_rank
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warm_up_proportion = warm_up_proportion
        self.no_cuda = no_cuda
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_mode = output_mode
        self.fp16 = fp16   
        self.discriminate = discriminate        
        self.gradual_unfreeze = gradual_unfreeze
        self.encoder_no = encoder_no


class FinBert(object):
    """
    The main class for FinBERT.
    """

    def __init__(self,
                 config):
        self.config = config

    def prepare_model(self, label_list):
        """
        Sets some of the components of the model: Dataset processor, number of labels, usage of gpu and distributed
        training, gradient accumulation steps and tokenizer.

        Parameters
        ----------
        label_list: list
            The list of labels values in the dataset. For example: ['positive','negative','neutral']
        """

        self.processors = {
            "finsent": FinSentProcessor
        }

        self.num_labels_task = {
            'finsent': 2
        }

        if self.config.local_rank == -1 or self.config.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.device, self.n_gpu, bool(self.config.local_rank != -1), self.config.fp16))

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config.gradient_accumulation_steps))

        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        if os.path.exists(self.config.model_dir) and os.listdir(self.config.model_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(self.config.model_dir))
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)


        self.processor = self.processors['finsent']()
        self.num_labels = len(label_list)
        self.label_list = label_list

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=self.config.do_lower_case)

    def get_data(self, phase):
        """
        Gets the data for training or evaluation. It returns the data in the format that pytorch will process. In the
        data directory, there should be a .csv file with the name <phase>.csv
        Parameters
        ----------
        phase: str
            Name of the dataset that will be used in that phase. For example if there is a 'train.csv' in the data
            folder, it should be set to 'train'.

        Returns
        -------
        examples: list
            A list of InputExample's. Each InputExample is an object that includes the information for each example;
            text, id, label...
        """

        self.num_train_optimization_steps = None
        examples = None
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(
                examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps) * self.config.num_train_epochs
        
        if phase=='train':
            train = pd.read_csv(os.path.join(self.config.data_dir, 'train.csv'),sep='\t',index_col=False)
            weights = list()
            labels = self.label_list
    
            class_weights = [train.shape[0] / train[train.label == label].shape[0] for label in labels]
            self.class_weights = torch.tensor(class_weights)

        return examples

    def create_the_model(self):
        """
        Creates the model. Sets the model to be trained and the optimizer.
        """

        model = self.config.bert_model

        model.to(self.device)

        # Prepare optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        lr = self.config.learning_rate
        dft_rate = 1.2

        if self.config.discriminate:
            # apply the discriminative fine-tuning. discrimination rate is governed by dft_rate.

            encoder_params = []
            for i in range(12):
                encoder_decay = {
                    'params': [p for n, p in list(model.bert.encoder.layer[i].named_parameters()) if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (12 - i))}
                encoder_nodecay = {
                    'params': [p for n, p in list(model.bert.encoder.layer[i].named_parameters()) if
                               any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr / (dft_rate ** (12 - i))}
                encoder_params.append(encoder_decay)
                encoder_params.append(encoder_nodecay)

            optimizer_grouped_parameters = [
                {'params': [p for n, p in list(model.bert.embeddings.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in list(model.bert.embeddings.named_parameters()) if
                            any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in list(model.bert.pooler.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr},
                {'params': [p for n, p in list(model.bert.pooler.named_parameters()) if
                            any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr},
                {'params': [p for n, p in list(model.classifier.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 'lr': lr},
                {'params': [p for n, p in list(model.classifier.named_parameters()) if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': lr}]

            optimizer_grouped_parameters.extend(encoder_params)


        else:
            param_optimizer = list(model.named_parameters())

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]


        schedule = "warmup_linear"



        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=self.config.learning_rate,
                                  warmup=self.config.warm_up_proportion,
                                  t_total=self.num_train_optimization_steps,
                                  schedule=schedule)

        return model

    def get_loader(self, examples, phase):
        """
        Creates a data loader object for a dataset.

        Parameters
        ----------
        examples: list
            The list of InputExample's.
        phase: 'train' or 'eval'
            Determines whether to use random sampling or sequential sampling depending on the phase.

        Returns
        -------
        dataloader: DataLoader
            The data loader object.
        """

        features = convert_examples_to_features(examples, self.label_list,
                                                self.config.max_seq_length,
                                                self.tokenizer,
                                                self.config.output_mode)

        # Log the necessasry information
        logger.info("***** Loading data *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)

        # Load the data, make it into TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if self.config.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.config.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        try:
            all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
        except:
            all_agree_ids = torch.tensor([0.0 for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_agree_ids)

        # Distributed, if necessary
        if phase == 'train':
            my_sampler = RandomSampler(data)
        elif phase == 'eval':
            my_sampler = SequentialSampler(data)

        dataloader = DataLoader(data, sampler=my_sampler, batch_size=self.config.train_batch_size)
        return dataloader

    def train(self, train_examples, model):
        """
        Trains the model.
        Parameters
        ----------
        examples: list
            Contains the data as a list of InputExample's
        model: BertModel
            The Bert model to be trained.
        weights: list
            Contains class weights.
        Returns
        -------
        model: BertModel
            The trained model.
        """
        
        
        validation_examples = self.get_data('validation')

        global_step = 0



        self.validation_losses = []

        # Training
        train_dataloader = self.get_loader(train_examples, 'train')

        model.train()

        step_number = len(train_dataloader)


        i = 0
        for _ in trange(int(self.config.num_train_epochs), desc="Epoch"):

            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):

                if (self.config.gradual_unfreeze and i == 0):
                    for param in model.bert.parameters():
                        param.requires_grad = False

                if (step % (step_number // 3)) == 0:
                    i += 1

                if (self.config.gradual_unfreeze and i > 1 and i < self.config.encoder_no):

                    for k in range(i - 1):

                        try:
                            for param in model.bert.encoder.layer[self.config.encoder_no - 1 - k].parameters():
                                param.requires_grad = True
                        except:
                            pass

                if (self.config.gradual_unfreeze and i > self.config.encoder_no + 1):
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = True

                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, agree_ids = batch

                logits = model(input_ids, segment_ids, input_mask)
                weights = self.class_weights.to(self.device)

                if self.config.output_mode == "classification":
                    loss_fct = CrossEntropyLoss(weight = weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.config.output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        lr_this_step = self.config.learning_rate * warmup_linear(
                            global_step / self.num_train_optimization_steps, self.config.warmup_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            # Validation

            validation_loader = self.get_loader(validation_examples, phase='eval')
            model.eval()

            valid_loss, valid_accuracy = 0, 0
            nb_valid_steps, nb_valid_examples = 0, 0

            for input_ids, input_mask, segment_ids, label_ids, agree_ids in tqdm(validation_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                agree_ids = agree_ids.to(self.device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)

                    if self.config.output_mode == "classification":
                        loss_fct = CrossEntropyLoss(weight=weights)
                        tmp_valid_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                    elif self.config.output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_valid_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    valid_loss += tmp_valid_loss.mean().item()

                    nb_valid_steps += 1

            valid_loss = valid_loss / nb_valid_steps

            self.validation_losses.append(valid_loss)
            print("Validation losses: {}".format(self.validation_losses))

            if valid_loss == min(self.validation_losses):

                try:
                    os.remove(self.config.model_dir / ('temporary' + str(best_model)))
                except:
                    print('No best model found')
                torch.save({'epoch': str(i), 'state_dict': model.state_dict()},
                           self.config.model_dir / ('temporary' + str(i)))
                best_model = i

        # Save a trained model and the associated configuration
        checkpoint = torch.load(self.config.model_dir / ('temporary' + str(best_model)))
        model.load_state_dict(checkpoint['state_dict'])
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(self.config.model_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.config.model_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        os.remove(self.config.model_dir / ('temporary' + str(best_model)))
        return model

    def evaluate(self, model, examples):
        """
        Evaluate the model.
        Parameters
        ----------
        model: BertModel
            The model to be evaluated.
        examples: list
            Evaluation data as a list of InputExample's/

        Returns
        -------
        evaluation_df: pd.DataFrame
            A dataframe that includes for each example predicted probability and labels.
        """

        eval_loader = self.get_loader(examples, phase='eval')

        logger.info("***** Running evaluation ***** ")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions = []
        labels = []
        agree_levels = []
        text_ids = []

        for input_ids, input_mask, segment_ids, label_ids, agree_ids in tqdm(eval_loader, desc="Testing"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            agree_ids = agree_ids.to(self.device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

                if self.config.output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                elif self.config.output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                np_logits = logits.cpu().numpy()

                if self.config.output_mode == 'classification':
                    prediction = np.array(np_logits)
                elif self.config.output_mode == "regression":
                    prediction = np.array(np_logits)

                for agree_id in agree_ids:
                    agree_levels.append(agree_id.item())

                for label_id in label_ids:
                    labels.append(label_id.item())

                for pred in prediction:
                    predictions.append(pred)

                text_ids.append(input_ids)

                # tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
                # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

            # logits = logits.detach().cpu().numpy()
            # label_ids = label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            # eval_accuracy += tmp_eval_accuracy

        evaluation_df = pd.DataFrame({'predictions': predictions, 'labels': labels, "agree_levels": agree_levels})

        return evaluation_df


def predict(text, model, write_to_csv=False, path=None):
    """
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.

    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    """
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sentences = sent_tokenize(text)

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence','logit','prediction','sentiment_score'])
    for batch in chunks(sentences, 5):

        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        with torch.no_grad():
            logits = model(all_input_ids, all_segment_ids, all_input_mask)
            logits = softmax(np.array(logits))
            sentiment_score = pd.Series(logits[:,0] - logits[:,1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {'sentence': batch,
                            'logit': list(logits),
                            'prediction': predictions,
                            'sentiment_score':sentiment_score}
            
            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result,batch_result], ignore_index=True)

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path,sep=',', index=False)

    return result

