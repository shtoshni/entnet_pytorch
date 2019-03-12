import torchtext.datasets as datasets
import torchtext

from os import path

# %% Download data
def get_data(data_dir):
    datasets.BABI20.download(data_dir)

# %% Calculate max size of story
def calc_memory_size(data_dir, task_id, tenK=True):
    root_dir = path.join(data_dir, "tasks_1-20_v1-2")
    if tenK:
        root_dir = path.join(root_dir, 'en-valid-10k')
    else:
        root_dir = path.join(root_dir, 'en-valid')

    task_train_file = path.join(root_dir, 'qa' + str(task_id) + '_train.txt')
    max_story_size = 0
    with open(task_train_file) as f:
        data = datasets.BABI20._parse(f, only_supporting=False)
        max_story_size = max([len(story) for story, _, _ in data])
    return max_story_size

# %% Get the iterators for train, validation, and test
def get_iter(data_dir, task_id=1, tenK=True, joint=False, memory_size=None):
    if memory_size is None:
        if task_id == 3:
            memory_size = 130
        elif joint:
            memory_size = 70
        else:
            memory_size = calc_memory_size(data_dir, task_id)
    print ("Max story size:", memory_size)
    train_iter, valid_iter, test_iter = datasets.BABI20.iters(batch_size=32,  root=data_dir, task=task_id, joint=joint, memory_size=memory_size, tenK=tenK, shuffle=True)
    return (train_iter, valid_iter, test_iter)


# %%
if __name__=='__main__':
    DATA_DIR = "/home/shtoshni/Research/entnet_pytorch/data"
    train_iter, valid_iter, test_iter = get_iter(DATA_DIR, joint=False, tenK=False, task_id=3)
    # print (len(train_iter.dataset.fields['answer'].vocab.freqs.keys()))
    # print (vars(train_iter.dataset.examples[0]).keys())

    # %%
    valid_data = valid_iter.data()
    itos = train_iter.dataset.fields['story'].vocab.itos
    print (itos)
    # print (len(itos))

    # print (len(train_iter.data()))
    # print (len(valid_data))
    # print (type(valid_data[0]))
    print ((valid_data[0].story))
    print ((valid_data[0].query))
    print ((valid_data[0].answer))

    for train_batch in train_iter:
        # print (vars(train_batch).keys())
        # print (len(train_batch.dataset.fields['answer'].vocab.freqs))
        # print (train_batch.dataset)
        # print (train_batch.target_fields)
        x_train, _ = train_batch
        stories, queries, answers = x_train
        print (type(stories))
        print (stories.size())
        print (answers.size())
        # print ((x_train[2][0][0].item()))
        # print (type(stories))
        break
