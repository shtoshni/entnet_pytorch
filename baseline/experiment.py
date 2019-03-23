import sys
import time
from os import path

import torch
import torch.nn as nn

from entnet import EntNet
from prep_data import get_iter, get_max_sentence_lengths
from utils import print_model_info

class Experiment:
    def __init__(self, data_dir, model_dir, best_model_dir,
                 # Data params
                 task_id, batch_size=32, joint=False, oneK=False,
                 # Model params
                 memory_slots=20, hidden_size=100, bow_encoding=False,
                 # Training params
                 rand_seed=0, init_lr=1e-2, lr_decay=25,
                 max_gradient_norm=40.0, max_epochs=200, eval=False):
        # Set the random seed first
        torch.manual_seed(rand_seed)

        # Prepare data info
        self.train_iter, self.valid_iter, self.test_iter = get_iter(data_dir,
         task_id=task_id, joint=joint, tenK=(not oneK), batch_size=batch_size)
        self.data_info = self.get_data_info()
        self.data_info['joint'] = joint

        # Get model paths
        self.model_path = path.join(model_dir, 'model_cur.pth')
        self.best_model_path = path.join(best_model_dir, 'model_best.pth')

        # Initialize model and training metadata
        self.initialize_setup(memory_slots=memory_slots,
                              hidden_size=hidden_size,
                              bow_encoding=bow_encoding,
                              init_lr=init_lr, lr_decay=lr_decay)

        if not eval:
            self.train(max_epochs=max_epochs,
                       max_gradient_norm=max_gradient_norm)
        # Finally evaluate model
        self.final_eval(model_dir)


    def get_data_info(self):
        """Get data info"""
        info = {}
        info['max_sentence_length'], info['max_query_length'] = get_max_sentence_lengths(self.train_iter)
        # Vocab size increased by 1 for PAD symbol
        info['vocab_size'] = len(self.train_iter.dataset.fields['answer'].vocab.freqs.keys()) + 1
        info['itos'] = self.train_iter.dataset.fields['story'].vocab.itos
        return info


    def initialize_setup(self, memory_slots, hidden_size, bow_encoding,
                         init_lr, lr_decay):
        """Initialize model and training info."""
        self.model = EntNet(vocab_size=self.data_info['vocab_size'], memory_slots=memory_slots, bow_encoding=bow_encoding, emb_size=hidden_size, max_sentence_length=self.data_info['max_sentence_length'], max_query_length=self.data_info['max_query_length']).cuda()

        print_model_info(self.model)

        self.train_info = {}
        self.train_info['optimizer'] = torch.optim.Adam(self.model.parameters(), lr=init_lr)
        self.train_info['lr_decay'] = lr_decay
        self.train_info['scheduler'] = torch.optim.lr_scheduler.StepLR(
            self.train_info['optimizer'], lr_decay, gamma=0.5)

        self.train_info['epochs_done'] = 0
        self.train_info['max_accuracy'] = 0.0

        # Check if a saved model already exists
        if path.exists(self.model_path):
            print ('Loading previous model')
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.train_info['optimizer'].load_state_dict(
                checkpoint['optimizer_state_dict'])
            self.train_info['scheduler'].load_state_dict(
                checkpoint['scheduler_state_dict'])
            self.train_info['epochs_done'] = checkpoint['epoch']
            self.train_info['max_accuracy'] = checkpoint['max_accuracy']


    def train(self, max_epochs, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epochs_done']
        optimizer = self.train_info['optimizer']
        scheduler = self.train_info['scheduler']

        for epoch in range(epochs_done, max_epochs):
            if self.train_info['max_accuracy'] == 1.0:
                print ("Reached max accuracy")
                return

            print ("Start Epoch", epoch + 1)
            # LR Scheduler step
            if self.train_info['lr_decay'] > 0:
                scheduler.step()
            # Start time counter
            start_time = time.time()
            # Model in train mode
            # self.train_iter.init_epoch()
            model.train()
            for i, train_batch in enumerate(self.train_iter):
                x_train, _ = train_batch
                stories, queries, answers = x_train
                # Move data to cuda
                stories, queries = stories.cuda(), queries.cuda()
                answers = torch.squeeze(answers).cuda()

                preds = model.read_story_and_answer_question(stories, queries)
                loss = model.get_loss(preds, answers)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                # Perform gradient clipping and update parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                optimizer.step()

                if i % 100 == 0:
                    print ("Step: %d, loss: %.4f" %(i, loss.item()))
                    # print (model.calc_gate_penalty(*additional_output).item())
                    sys.stdout.flush()

            # Validation accuracy
            accuracy = self.eval_model()
            # Update epochs done
            self.train_info['epochs_done'] = epoch + 1

            # Update model if validation performance improves
            if accuracy > self.train_info['max_accuracy']:
                self.train_info['max_accuracy'] = accuracy
                print ('Saving best model')
                self.save_model(self.best_model_path)
            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            print ("Epoch: %d, Time: %.2f, Accuracy: %.3f (Max: %.3f)" %(
                epoch + 1, elapsed_time, accuracy,
                self.train_info['max_accuracy']))
            sys.stdout.flush()


    def eval_model(self, data_iter=None):
        """Eval model"""
        self.model.eval()
        total = 0.0
        correct = 0.0

        data_iter = (data_iter if data_iter else self.valid_iter)
        for j, batch_data in enumerate(data_iter):
            x_data, _ = batch_data
            stories, queries, answers = x_data
            # Move data to GPU
            stories, queries = stories.cuda(), queries.cuda()
            answers = torch.squeeze(answers).cuda()

            preds = self.model.read_story_and_answer_question(stories, queries)
            max_indices = torch.argmax(preds, 1)
            correct += torch.sum(max_indices == answers).item()
            total += answers.size()[0]

        return correct/total


    def final_eval(self, model_dir):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        valid_accuracy = checkpoint['max_accuracy']
        train_accuracy = self.eval_model(self.train_iter)

        test_accuracy = []
        if self.data_info['joint']:
            for task_id in range(1, 21):
                test_accuracy.append(self.eval_model(self.test_iter[task_id]))
        else:
            test_accuracy.append(self.eval_model(self.test_iter))

        print ("Train accuracy: %.3f" %train_accuracy)
        print ("Validation accuracy: %.3f" %valid_accuracy)

        avg_test_accuracy = sum(test_accuracy)/len(test_accuracy)
        print ("Test accuracy: %.3f" %avg_test_accuracy)
        if self.data_info['joint']:
            print ("Test accuracy for all tasks: ", test_accuracy)

        perf_file = path.join(model_dir, 'perf.txt')
        print ('Performance output at:', perf_file)
        with open(perf_file, 'w') as f:
            f.write("%.3f\n%.3f\n%.3f\n" %(train_accuracy, valid_accuracy, avg_test_accuracy))
        sys.stdout.flush()


    def save_model(self, location):
        """Save model"""
        torch.save({
            'epoch': self.train_info['epochs_done'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.train_info['optimizer'].state_dict(),
            'scheduler_state_dict': self.train_info['scheduler'].state_dict(),
            'max_accuracy': self.train_info['max_accuracy'],
            'itos': self.data_info['itos'],
        }, location)
        print ("Model saved at:", location)
