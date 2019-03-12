import time
from os import path

import torch
import torch.nn as nn

from entnet import EntNet
from prep_data import get_iter

def do_experiment(data_dir, model_dir, best_model_dir, task_id, batch_size=32, joint=False, memory_slots=20, hidden_size=100, bow_encoding=False, rand_seed=0, gate_penalty=0.0, init_lr=1e-2, lr_decay=25, max_gradient_norm=40.0, max_sentences=None, max_epochs=200):
    """Do an experiment with the given attributes."""
    torch.manual_seed(rand_seed)
    train_iter, valid_iter, test_iter = get_iter(data_dir, task_id=task_id, joint=joint, memory_size=max_sentences)

    # Get model paths
    model_path = path.join(model_dir, 'model_cur.pth')
    best_model_path = path.join(best_model_dir, 'model_best.pth')

    # Vocab size increased by 1 for PAD symbol
    vocab_size = len(train_iter.dataset.fields['answer'].vocab.freqs.keys()) + 1
    itos = train_iter.dataset.fields['story'].vocab.itos

    model = EntNet(vocab_size=vocab_size, memory_slots=memory_slots, bow_encoding=bow_encoding, emb_size=hidden_size, gate_penalty=gate_penalty).cuda()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay, gamma=0.5)

    # Initialize variables for tracking progress
    epochs_done = 0
    max_accuracy = 0.0

    # Check if a saved model already exists
    if path.exists(model_path):
        print ('Loading previous model')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epochs_done = checkpoint['epoch']
        max_accuracy = checkpoint['max_accuracy']

    for epoch in range(epochs_done, max_epochs):
        print ("Start Epoch", epoch)
        # LR Scheduler step
        if lr_decay > 0:
            scheduler.step()
        # Start time counter
        start_time = time.time()
        # Model in train mode
        train_iter.init_epoch()
        model.train()
        for i, train_batch in enumerate(train_iter):
            x_train, _ = train_batch
            stories, queries, answers = x_train
            # if i == 0:
            #     print (answers)
            # print (stories.type())
            stories = stories.cuda()
            queries = queries.cuda()
            answers = torch.squeeze(answers).cuda()
            # print (answers.size())

            preds, gate_list = model.read_story_and_answer_question(stories, queries)
            loss = model.get_loss(preds, answers, gate_list)

            optimizer.zero_grad()
            loss.backward()

            # Perform gradient clipping and update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()

            if i % 100 == 0:
                print ("Step: %d, loss: %.4f" %(i, loss.item()))
                # break

        model.eval()
        total = 0.0
        correct = 0.0
        for j, valid_batch in enumerate(valid_iter):
            x_valid, _ = valid_batch
            stories, queries, answers = x_valid
            stories = stories.cuda()
            queries = queries.cuda()
            answers = torch.squeeze(answers).cuda()

            preds, _ = model.read_story_and_answer_question(stories, queries)
            max_indices = torch.argmax(preds, 1)

            correct += torch.sum(max_indices == answers).item()
            total += answers.size()[0]

        # Validation accuracy
        accuracy = correct/total

        # Update gate penalty
        if (epoch + 1) % 10 == 0:
            model.update_gate_penalty()

        # Save model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'max_accuracy': max_accuracy,
            'itos': itos,
        }, model_path)

        # Update the improved model
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            print ('Saving best model')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'max_accuracy': max_accuracy,
                'itos': itos,
            }, best_model_path)

        # Get elapsed time
        elapsed_time = time.time() - start_time
        print ("Epoch: %d, Time: %.2f, Accuracy: %.3f (Max: %.3f)" %(epoch, elapsed_time, accuracy, max_accuracy))
