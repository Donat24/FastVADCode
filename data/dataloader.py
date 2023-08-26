from util.datasets import *
from .data_other import *
from .data_selfmade import *
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

#Costume Collate
def costume_collate_fn(batch):

    x_list = []
    y_list = []

    #Iter
    for x, y in batch:
        x_list.append(x)
        y_list.append(y)

    #Padding
    x = torch.nn.utils.rnn.pad_sequence(sequences = x_list, batch_first=True, padding_value=0)

    #Stack Y
    y = torch.nn.utils.rnn.pad_sequence(sequences = y_list, batch_first=True, padding_value=0)

    return x, y

#TRAIN DATA
my_train,_                 = random_split(dataset_train,                 lengths=[60000, len(dataset_train)                - 60000])
speech_commands,_          = random_split(dataset_speechcommands,        lengths=[10000, len(dataset_speechcommands)       - 10000])
common_voice_german,_      = random_split(dataset_common_voice_german,   lengths=[10000, len(dataset_common_voice_german)  - 10000])
common_voice_english,_      = random_split(dataset_common_voice_english, lengths=[10000, len(dataset_common_voice_english) - 10000])
ava_train, dataset_ava_val = random_split(dataset_ava_train_splitted,    lengths=[10000, len(dataset_ava_train_splitted)   - 10000])

#VAL DATA
MY_VAL_SAMPLES  = 900
AVA_VAL_SAMPLES = 1000
LIBRIARTY_VAL_SAMPLES = 220

my_val, _        = random_split(dataset_val,            lengths=[MY_VAL_SAMPLES,        len(dataset_val)            - MY_VAL_SAMPLES])
ava_val,_        = random_split(dataset_ava_val,        lengths=[AVA_VAL_SAMPLES,       len(dataset_ava_val)        - AVA_VAL_SAMPLES])
libriParty_val,_ = random_split(dataset_LibriParty_val, lengths=[LIBRIARTY_VAL_SAMPLES, len(dataset_LibriParty_val) - LIBRIARTY_VAL_SAMPLES])


train = ConcatDataset([my_train, speech_commands, common_voice_german, common_voice_english, ava_train])
#val   = ConcatDataset([my_val,   ava_val, libriParty_val])
val   = ConcatDataset([my_val,   ava_val])

#Dataloader für Training
dataloader_train = DataLoader( train, batch_size=BATCH_SIZE,     shuffle=True,  collate_fn=costume_collate_fn, pin_memory=False, num_workers=16 )
dataloader_val   = DataLoader( val,   batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=costume_collate_fn, pin_memory=False, num_workers=16 )
#dataloader_test  = DataLoader( dataset_test,  batch_size=1, shuffle=False, collate_fn=costume_collate_fn, pin_memory=False, num_workers=0 )