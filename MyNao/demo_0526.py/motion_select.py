import numpy as np
import pickle
import sys
import time

class Motion_Selector(object):
    def __init__(self, current=0):
        with open('Amendments/scipts for graph construction/graph construction 0522/03.stable_typical_sequences_symmetrical.txt', 'rb') as f:
            self.stable_typical_sequence = pickle.load(f)
        self.sequence_head = []
        self.sequence_tail = []
        for sequence in self.stable_typical_sequence:
            self.sequence_head.append(sequence[0])
            self.sequence_tail.append(sequence[-1])
        with open('Amendments/scipts for graph construction/graph construction 0522/04.nx5_matrix_and_interface.txt', 'rb') as f:
            self.nx5_matrix = pickle.load(f)
            self.interface = pickle.load(f)
        with open('Amendments/scipts for graph construction/graph construction 0522/06.transition_graph.txt', 'rb') as f:
            self.limited_transition_graph = pickle.load(f)
            self.inter_transition_graph = pickle.load(f)
        
        self.current = current
        if self.current in self.sequence_head:
            self.mode = 1
        elif self.current not in self.stable_typical_sequence:
            self.mode = 0
        else:
            print('Reassign the initial dance motion primitive!')
            sys.exit()
        self.row = 0
        self.column = 0
        self.count = 0
        time.sleep(1)


    def transfer(self):
        if self.mode == 0:
            if self.count < 3:
                self.count += 1
                self.current = np.random.choice(self.limited_transition_graph[self.current]['next'], p=np.array(self.limited_transition_graph[self.current]['p']).ravel())
                return self.current
            else:
                self.count =0
                self.current = np.random.choice(self.inter_transition_graph[self.current]['next'], p=np.array(self.inter_transition_graph[self.current]['p']).ravel())
                if self.current in self.sequence_head:
                    self.mode = 1
                    p = np.array(self.sequence_head==self.current)
                    self.row = np.random.choice(self.interface, p=(p/np.sum(p)).ravel())
                    return self.current
        if self.mode == 1:
            if self.nx5_matrix[self.row][self.column+1] == -1:
                self.column = 0
                self.current = np.random.choice(self.inter_transition_graph[self.current]['next'], p=np.array(self.inter_transition_graph[self.current]['p']).ravel())
                if not self.current in self.sequence_head:
                    self.mode = 0
                return self.current
            self.column += 1
            if self.column == 4:
                self.row += 1
                self.column = 0
            self.current = self.nx5_matrix[self.row][self.column]
            return self.current

    def transmit_motion_info(self, queue):
        if not queue.empty():
            _ = queue.get()
        queue.put(self.current)
        #print('mode:', self.mode, 'current:', self.current)

    def show(self):
        print('mode:', self.mode, 'current:', self.current)

if __name__ == '__main__':
    motion_selector = Motion_Selector(0)
    for i in range(20):
        motion_selector.show()
        current = motion_selector.transfer()
